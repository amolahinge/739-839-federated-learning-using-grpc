from __future__ import print_function
from inspect import getmembers

import logging
import argparse

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

clients=[]
compressFlag = False
optimModelPath = "optimizedModel.pth"
backupServerChannel = None
isPrimaryUp = 1
isRunningAsPrimaryServer = 0
recovering = 1

isPrimary = True
options = [
    ('grpc.max_send_message_length', 1024 * 1024 * 1024),
    ('grpc.max_receive_message_length', 1024 * 1024 * 1024)
]

def getMountedPath(suffixPath):
    return mountPoint + '/'+ suffixPath

def trainThreadFunc(count,channel):
    net = MobileNet()
    stub = federated_pb2_grpc.TrainerStub(channel)
    response = stub.StartTrain(federated_pb2.TrainRequest(name='you',rank=count,world=len(clients)))
    model=base64.b64decode(response.message)
    f = open(getMountedPath("test_"+str(count)+".pth"),'wb')
    f.write(model)
    f.close()

def sendOptimizedModel(channel):        # TODO - With optimized model send epoch no. to server if want to resume
    f=open(getMountedPath(optimModelPath), "rb")
    encode=base64.b64encode(f.read())
    f.close()
    stub = federated_pb2_grpc.TrainerStub(channel)
    response = stub.SendModel(federated_pb2.SendModelRequest(model=encode))

def run():
    channels=[]
    count=0
    # threads=[]
    for client in clients:
        if compressFlag:
            channels.append(grpc.insecure_channel(client,options=options,compression=grpc.Compression.Gzip))
        else:
            channels.append(grpc.insecure_channel(client,options=options))
    for epoch in range(4):
        net = MobileNet()
        trainthreads=[]
        for count,channel in enumerate(channels):
            trainthreads.append(threading.Thread(target=trainThreadFunc, args=(count,channel)))
        for i in range(len(trainthreads)):
            trainthreads[i].start()
        for i in range (len(trainthreads)):
            trainthreads[i].join()
        allreduce()
        
        sendThreads=[]
        if backupServerChannel is not None:
            sendOptimizedModel(backupServerChannel)
        for count,channel in enumerate(channels):
            sendThreads.append(threading.Thread(target=sendOptimizedModel, args=(channel,)))
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

    # print("Before summing", pushedModels[0].state_dict())
    optimizedModel = pushedModels[0].state_dict()
    
    for index in range(1, len(pushedModels)):
        m = pushedModels[index].state_dict()
        for key in m:
            optimizedModel[key] = optimizedModel[key] + m[key]
    # print("After summing", optimizedModel)

    for key in optimizedModel:
        optimizedModel[key] = optimizedModel[key] / len(pushedModels)

    # print("After averaging", optimizedModel)

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
            time.sleep(5)
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

    clients.append('localhost:50051')
    #clients.append('localhost:50052')

    if args.p == 'y':
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