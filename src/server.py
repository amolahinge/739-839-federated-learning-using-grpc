from __future__ import print_function
from inspect import getmembers

import logging

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

clients=[]
optimModelPath = "optimizedModel.pth"
backupServerChannel = None

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

def sendOptimizedModel(channel):
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
        channels.append(grpc.insecure_channel(client,options=options))
    for epoch in range(2):
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

def ConnectToBackupServer(altaddress, port):
    addr = altaddress + ":" + port
    global backupServerChannel
    backupServerChannel = grpc.insecure_channel(addr, options=options)

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
# Functions related to Backup server only
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),options = [
        ('grpc.max_send_message_length', 1024 * 1024 * 1024),
        ('grpc.max_receive_message_length', 1024 * 1024 * 1024)
    ])
    federated_pb2_grpc.add_TrainerServicer_to_server(Trainer(), server)
    server.add_insecure_port('[::]:8080')
    server.start()
    server.wait_for_termination()

class Trainer(federated_pb2_grpc.TrainerServicer):
    def SendModel(self,request,context):
        print("This function started running")
        f = open(getMountedPath(optimModelPath), "wb")
        decode = base64.b64decode(request.model)
        f.write(decode)
        f.close()
        return federated_pb2.SendModelReply(reply = "success")

#######################################################################

if __name__ == '__main__':
    logging.basicConfig()
    clients.append('localhost:50051')
    #clients.append('localhost:50052')

    parser = argparse.ArgumentParser(description='Server Information')
    parser.add_argument('--p', default='n', help='Is Primary?')
    parser.add_argument('--altAddress', default='', help='Backup Server address')
    parser.add_argument('--port', default='', help='Port')
    args = parser.parse_args()

    if args.p == 'y':
        print("Primary triggered")
        ConnectToBackupServer(args.altAddress, args.port)
        mountPoint = "Primary"
        Path(mountPoint).mkdir(parents=True, exist_ok=True)
        run()
    else:
        print("Backup triggered")
        mountPoint = "Backup"
        Path(mountPoint).mkdir(parents=True, exist_ok=True)
        serve()

