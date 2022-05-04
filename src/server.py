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


channels={} #client - client channel
clients={} # client - active / inactive status
compressFlag = False
optimModelPath = "optimizedModel.pth"
currentEpochPath = "currentEpoch.txt"
backupServerChannel = None
isPrimaryUp = 1
isRunningAsPrimaryServer = 0
recovering = 0
clientTracking = None
stop_thread = False

isPrimary = True
options = [
    ('grpc.max_send_message_length', 1024 * 1024 * 1024),
    ('grpc.max_receive_message_length', 1024 * 1024 * 1024)
]

def getMountedPath(suffixPath):
    return mountPoint + '/'+ suffixPath

def writeIntegerToFile(number):
    with open(getMountedPath(currentEpochPath), 'w') as f:
        f.write('%d' % number)

def readIntegerFromFile():
    my_file = Path(getMountedPath(currentEpochPath))
    if not my_file.is_file():
        return 1
    f = open(getMountedPath(currentEpochPath), 'r')
    currIndex = f.readline()
    f.close()
    return int(currIndex) + 1


def trainThreadFunc(epoch, count, client, channel):
    stub = federated_pb2_grpc.TrainerStub(channel)
    try:
        response = stub.StartTrain(federated_pb2.TrainRequest(epoch = epoch, rank=count,world=len(clients)))
        model=base64.b64decode(response.message)
        f = open(getMountedPath("test_"+str(count)+".pth"),'wb')
        f.write(model)
        f.close()
    except grpc.RpcError as e:
        status_code = e.code()
        #if grpc.StatusCode.UNAVAILABLE == status_code:
        clients[client] = False

def sendOptimizedModel(epoch, client, channel):        # TODO - With optimized model send epoch no. to server if want to resume

    f=open(getMountedPath(optimModelPath), "rb")
    encode=base64.b64encode(f.read())
    f.close()
    try:
        stub = federated_pb2_grpc.TrainerStub(channel)
        response = stub.SendModel(federated_pb2.SendModelRequest(model = encode, epoch = epoch))
    except grpc.RpcError as e:
        status_code = e.code()
        #if grpc.StatusCode.UNAVAILABLE == status_code:
        clients[client] = False            
            #del channels[client] 

def checkClientStatus():
    #logic to check if we are able to re-establish connection with client
    while(True and not stop_thread):
        
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
                        sendOptimizedModel(0, key,channel)  # Sending dummy value for epoch. For client, epoch no. used from train
                except grpc.RpcError as e:
                    status_code = e.code()
                    #print("GRPC Error", status_code)
        print("Client status", clients)
        time.sleep(20)

def createChannel(client):
    if compressFlag:
        return grpc.insecure_channel(client,options=options,compression=grpc.Compression.Gzip)
    else:
        return grpc.insecure_channel(client,options=options)

def init():
    for client in clients.keys():
        channels[client] = createChannel(client)

def run(initial):
    init()
    #fault tolerance
    global clientTracking
    clientTracking = threading.Thread(target=checkClientStatus)
    clientTracking.start()

    for epoch in range(initial, 20):
        print("Starting epoch", epoch)
        count=0
        # threads=[]        
        trainthreads=[]
        #print(len(channels))
        for client,channel in channels.items():
            if clients[client]:
                trainthreads.append(threading.Thread(target=trainThreadFunc, args=(epoch, count, client, channel)))
                count = count + 1
        #print("Train thread count", count)

        for i in range(len(trainthreads)):
            trainthreads[i].start()
        for i in range (len(trainthreads)):
            trainthreads[i].join()
        allreduce()
        
        sendThreads=[]

        count = 0
        print("Replicating Global Model to Backup server")

        if backupServerChannel is not None:
            sendOptimizedModel(epoch, None, backupServerChannel)
        
        print("Sending Global Model to all Clients")
        for client,channel in channels.items():
            if clients[client]:
                sendThreads.append(threading.Thread(target=sendOptimizedModel, args=(epoch, client, channel)))
                count = count + 1
        #print("Send updated model thread count", count)

        writeIntegerToFile(epoch)

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

def fetchingModelAndEpoch():
    global recovering
    stub = federated_pb2_grpc.TrainerStub(backupServerChannel)
    
    try:
        #print("Recovering", recovering)
        if recovering == 1:
            response = stub.ReceiveModel(federated_pb2.Request())
            f = open(getMountedPath(optimModelPath), "wb")
            decode = base64.b64decode(response.model)
            f.write(decode)
            f.close()
            print("Epoch number received = ", response.epoch)
            writeIntegerToFile(response.epoch)
        #this call is being used by backup to identify if primary is up or not
        response = stub.CheckIfPrimaryUp(federated_pb2.PingRequest(req = str(recovering)))
        recovering = 0
    except Exception as e:
        #print(e)
        print("Primary not able to connect to backup server")

def pingBackupServer():
    while(1):
        fetchingModelAndEpoch()
        print("Heartbeat Sent")
        time.sleep(3)

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
    #print("isRunningAsPrimaryServer :", isRunningAsPrimaryServer)
    if isRunningAsPrimaryServer == 0:
        print("Backup server started running as Primary")
        isRunningAsPrimaryServer = 1
        run(readIntegerFromFile())
    else:
        print("Backup server started running as Backup again")
        isRunningAsPrimaryServer = 0
        global stop_thread
        if clientTracking is not None:
            stop_thread = True
            clientTracking.join()    # Thread not terminating
        stop_thread = False
        t = threading.Thread(target = CheckingIfPrimaryServerUp)
        t.start()
        serve(args.backupPort)

class Trainer(federated_pb2_grpc.TrainerServicer):
    def SendModel(self,request,context):
        #print("This function started running")
        f = open(getMountedPath(optimModelPath), "wb")
        decode = base64.b64decode(request.model)
        f.write(decode)
        f.close()
        writeIntegerToFile(request.epoch)
        print("Replicated Global Model")
        return federated_pb2.SendModelReply(reply = "success")

    def ReceiveModel(self,request,context):
        f=open(getMountedPath(optimModelPath), "rb")
        encode=base64.b64encode(f.read())
        f.close()
        print("Epoch number sent is ", readIntegerFromFile())
        return federated_pb2.ReceiveModelResponse(model = encode, epoch = readIntegerFromFile())

    def CheckIfPrimaryUp(self, request, context):
        print("Heartbeat Acknowledged")
        global isPrimaryUp
        isPrimaryUp = 1
        # Check if primary is recovering and if backup server was running as primary or not
        if(request.req == "1" and isRunningAsPrimaryServer == 1):
            print()
            print("The backup server was running as Primary server. Therefore switching")
            os.kill(os.getpid(), signal.SIGUSR1)
        return federated_pb2.PingResponse(value = 1)

def CheckingIfPrimaryServerUp():
    global isPrimaryUp
    flag = True
    while(flag):
        if isPrimaryUp == 1:
            isPrimaryUp = 0
            time.sleep(8)
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
    parser.add_argument('--recover', default='n', help='Backup Server Port')

    args = parser.parse_args()
    if args.compressFlag == "Y": 
        compressFlag = True        
    print("Compression {} enabled".format(args.compressFlag))
    
    clients['localhost:50051'] = True
    clients['localhost:50052'] = True
    #clients.append('localhost:50053')
    #clients.append('localhost:50054')

    if args.p == 'y':
        print("Primary triggered")
        ConnectToBackupServer(args.backupAddress, args.backupPort)
        mountPoint = "Primary"
        Path(mountPoint).mkdir(parents=True, exist_ok=True)
        if args.recover == 'n' and Path(getMountedPath(currentEpochPath)).is_file():
            os.remove(getMountedPath(currentEpochPath))
        if args.recover == 'y':
            recovering = 1
        while (recovering == 1):
            fetchingModelAndEpoch()
        t = threading.Thread(target = pingBackupServer)
        t.start()
        run(readIntegerFromFile())
    else:
        print("Backup triggered")
        mountPoint = "Backup"
        Path(mountPoint).mkdir(parents=True, exist_ok=True)
        signal.signal(signal.SIGUSR1, handler)
        if Path(getMountedPath(currentEpochPath)).is_file():
            os.remove(getMountedPath(currentEpochPath))
        if Path(getMountedPath(optimModelPath)).is_file():
            os.remove(getMountedPath(optimModelPath))
        t = threading.Thread(target = CheckingIfPrimaryServerUp)
        t.start()
        serve(args.backupPort)
