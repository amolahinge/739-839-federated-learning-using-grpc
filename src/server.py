from __future__ import print_function
from inspect import getmembers

import logging
import argparse
import time
import copy

from concurrent import futures
import argparse
import grpc
import federated_pb2
import federated_pb2_grpc
import base64
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import threading
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from models import *
import time
import os, signal
import sys
import pickle
from datetime import datetime

channels={} #client - client channel
clients={} # client - active / inactive status
compressFlag = False
optimModelPath = "optimizedModel.pth"
backupServerChannel = None
isPrimaryUp = 1
isRunningAsPrimaryServer = 0
recovering = 1
clientTracking = None

isPrimary = True
options = [
    ('grpc.max_send_message_length', 1024 * 1024 * 1024),
    ('grpc.max_receive_message_length', 1024 * 1024 * 1024)
]

def getMountedPath(suffixPath):
    return mountPoint + '/'+ suffixPath


def trainThreadFunc(count, client, channel):
    stub = federated_pb2_grpc.TrainerStub(channel)
    try:
        print("Starting train",datetime.now())
        response = stub.StartTrain(federated_pb2.TrainRequest(rank=count,world=len(clients)))
        print("Got model",datetime.now())

        print("pickle dump",len(pickle.dumps(response)))
        print("response ",sys.getsizeof(response))
        print("before compression",sys.getsizeof(response.message))

        model=base64.b64decode(response.message)
        print("Decoding complete",datetime.now())

        print("after compression",sys.getsizeof(model))
        f = open(getMountedPath("test_"+str(count)+".pth"),'wb')
        f.write(model)
        f.close()
        print("File saved",datetime.now())

    except grpc.RpcError as e:
        status_code = e.code()
        #if grpc.StatusCode.UNAVAILABLE == status_code:
        clients[client] = False

def sendOptimizedModel(client, channel):        # TODO - With optimized model send epoch no. to server if want to resume
    
    f=open(getMountedPath(optimModelPath), "rb")
    encode=base64.b64encode(f.read())
    f.close()
    try:
        stub = federated_pb2_grpc.TrainerStub(channel)
        response = stub.SendModel(federated_pb2.SendModelRequest(model=encode))
    except grpc.RpcError as e:
        status_code = e.code()
        #if grpc.StatusCode.UNAVAILABLE == status_code:
        clients[client] = False            
            #del channels[client] 

def checkClientStatus():
    #logic to check if we are able to re-establish connection with client
    while(True):
        time.sleep(1)
        clients_copy = copy.copy(clients)
        for key in clients_copy.keys():
            #iterate over inactive clients only
            if not clients_copy[key]:
                try:
                    #creating new channel here
                    channel = createChannel(key)
                    stub = federated_pb2_grpc.TrainerStub(channel)
                    response = stub.HeartBeat(federated_pb2.Request())
                    #if there is no error here
                    if response.status == 1:
                        clients[key] = True
                        channels[key] = channel
                        #send updated model again
                        print("Sending updated model with count value", list(clients_copy.keys()).index(key))
                        sendOptimizedModel(key,channel)
                except grpc.RpcError as e:
                    status_code = e.code()
                    #print("GRPC Error", status_code)
        print("Client status", clients)

def createChannel(client):
    if compressFlag:
        return grpc.insecure_channel(client,options=options,compression=grpc.Compression.Deflate)
    else:
        return grpc.insecure_channel(client,options=options)

def init():
    for client in clients.keys():
        channels[client] = createChannel(client)

def run():
    init()
    #fault tolerance
    global clientTracking
    clientTracking = threading.Thread(target=checkClientStatus)
    clientTracking.start()

    for epoch in range(1):
        print("Starting epoch", epoch)
        count=0
        # threads=[]        
        trainthreads=[]
        #print(len(channels))
        for client,channel in channels.items():
            if clients[client]:
                trainthreads.append(threading.Thread(target=trainThreadFunc, args=(count,client,channel)))
                count = count + 1
        print("Train thread count", count)

        for i in range(len(trainthreads)):
            trainthreads[i].start()
        for i in range (len(trainthreads)):
            trainthreads[i].join()
        allreduce()
        
        sendThreads=[]

        count = 0
        if backupServerChannel is not None:
            sendOptimizedModel(None, backupServerChannel)

        for client,channel in channels.items():
            if clients[client]:
                sendThreads.append(threading.Thread(target=sendOptimizedModel, args=(client, channel)))
                count = count + 1
        print("Send updated model thread count", count)

        for i in range(len(sendThreads)):
            sendThreads[i].start()
        for i in range (len(sendThreads)):
            sendThreads[i].join()

def allreduce():
    pushedModels = []
    for i in range(0, len(clients)):
        m = MobileNet()
        path = getMountedPath("test_"+str(i)+".pth")
        m.load_state_dict(torch.load(path)['net'])
        pushedModels.append(m)

    optimizedModel = pushedModels[0].state_dict()
    
    for index in range(1, len(pushedModels)):
        m = pushedModels[index].state_dict()
        for key in m:
            optimizedModel[key] = optimizedModel[key] + m[key]

    for key in optimizedModel:
        optimizedModel[key] = optimizedModel[key] / len(pushedModels)


    state = {
        'net': optimizedModel,
        'acc': 1,
        'epoch': 1,
    }
    torch.save(state, getMountedPath(optimModelPath))

#######################################################################
# Functions related to Primary server only
def ConnectToBackupServer(altaddress, port):
    addr = altaddress + ":" + port
    global backupServerChannel
    backupServerChannel = grpc.insecure_channel(addr, options=options)

def pingBackupServer():
    global recovering
    stub = federated_pb2_grpc.TrainerStub(backupServerChannel)
    while(1):
        try:
            print("Recovering", recovering)
            #this call is being used by backup to identify if primary is up or not
            response = stub.CheckIfPrimaryUp(federated_pb2.PingRequest(req = str(recovering)))
            recovering = 0
        except Exception:
            recovering = 0
            print("Happens")
        time.sleep(1)

#######################################################################



#######################################################################
# Functions related to Backup server only
def serve(port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),options = [
        ('grpc.max_send_message_length', 1024 * 1024 * 1024),
        ('grpc.max_receive_message_length', 1024 * 1024 * 1024)
    ])
    federated_pb2_grpc.add_TrainerServicer_to_server(Trainer(), server)
    server.add_insecure_port('[::]:' + port)
    server.start()
    server.wait_for_termination()

# Whenever the process is interrupted, this function is called
def handler(signum, frame):
    global isRunningAsPrimaryServer
    print("isRunningAsPrimaryServer :", isRunningAsPrimaryServer)
    if isRunningAsPrimaryServer == 0:
        print("Start running as Primary server")
        isRunningAsPrimaryServer = 1
        run()
    else:
        print("Start running as Backup server")
        isRunningAsPrimaryServer = 0
        if clientTracking is not None:
            clientTracking.terminate()
        t = threading.Thread(target = CheckingIfPrimaryServerUp)
        t.start()
        serve(args.backupPort)

class Trainer(federated_pb2_grpc.TrainerServicer):
    def SendModel(self,request,context):
        print("This function started running")
        f = open(getMountedPath(optimModelPath), "wb")
        decode = base64.b64decode(request.model)
        f.write(decode)
        f.close()
        return federated_pb2.SendModelReply(reply = "success")

    def CheckIfPrimaryUp(self, request, context):
        print("Ping received")
        global isPrimaryUp
        isPrimaryUp = 1
        # Check if primary is recovering and if backup server was running as primary or not
        if(request.req == "1" and isRunningAsPrimaryServer == 1):
            print("The backup server was running as Primary server. Therefore switching")
            os.kill(os.getpid(), signal.SIGUSR1)
        return federated_pb2.PingResponse(value = 1)

def CheckingIfPrimaryServerUp():
    global isPrimaryUp
    flag = True
    while(flag):
        if isPrimaryUp == 1:
            isPrimaryUp = 0
            time.sleep(10)
        else:
            print("Got to know primary server is down")
            os.kill(os.getpid(), signal.SIGUSR1)
            flag = False

#######################################################################

if __name__ == '__main__':
    logging.basicConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--compressFlag", help="Compression enabled/disabled")
    parser.add_argument('--p', default='n', help='Is Primary?')
    parser.add_argument('--backupAddress', default='localhost', help='Backup Server address')
    parser.add_argument('--backupPort', default='8080', help='Backup Server Port')

    args = parser.parse_args()
    if args.compressFlag == "Y": 
        compressFlag = True        
    print("Compression {} enabled".format(args.compressFlag))
    
    clients['localhost:50051'] = True
  #  clients['localhost:50052'] = True
    #clients.append('localhost:50053')
    #clients.append('localhost:50054')

    if args.p != 'y':
        print("Primary triggered")
        ConnectToBackupServer(args.backupAddress, args.backupPort)
        mountPoint = "Primary"
        Path(mountPoint).mkdir(parents=True, exist_ok=True)
        t = threading.Thread(target = pingBackupServer)
        t.start()
        run()
    else:
        print("Backup triggered")
        mountPoint = "Backup"
        Path(mountPoint).mkdir(parents=True, exist_ok=True)
        signal.signal(signal.SIGUSR1, handler)
        t = threading.Thread(target = CheckingIfPrimaryServerUp)
        t.start()
        serve(args.backupPort)
