from __future__ import print_function

import logging
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
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from models import *

clients=[]
compressFlag = False

def trainThreadFunc(count,channel):
    net = MobileNet()
    stub = federated_pb2_grpc.TrainerStub(channel)
    response = stub.StartTrain(federated_pb2.TrainRequest(name='you',rank=count,world=len(clients)))
    model=base64.b64decode(response.message)
    f = open("test_"+str(count)+".pth",'wb')
    f.write(model)
    f.close()

def sendThreadFunc(count,channel):
    optimmodel="./checkpoint/ckpt.pth"
    f=open(optimmodel, "rb")
    encode=base64.b64encode(f.read())
    f.close()
    stub = federated_pb2_grpc.TrainerStub(channel)
    response = stub.SendModel(federated_pb2.SendModelRequest(model=encode))

def run():
    options = [
        ('grpc.max_send_message_length', 1024 * 1024 * 1024),
        ('grpc.max_receive_message_length', 1024 * 1024 * 1024)
    ]

    channels=[]
    count=0
    # threads=[]
    for client in clients:
        if compressFlag:
            channels.append(grpc.insecure_channel(client,options=options,compression=grpc.Compression.Gzip))
        else:
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
        #allreduce()
        optimmodel="optimizedModel.pth"
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        optimmodel="./checkpoint/ckpt.pth"

        # checkpoint = torch.load('optimizedModel.pth')
        # net.load_state_dict(checkpoint['net'])
        sendThreads=[]
        for count,channel in enumerate(channels):
            sendThreads.append(threading.Thread(target=sendThreadFunc, args=(count,channel)))
        for i in range(len(sendThreads)):
            sendThreads[i].start()
        for i in range (len(sendThreads)):
            sendThreads[i].join()

def allreduce():
    pushedModels = []
    for i in range(0, 3):   # TODO - This should be a configurable parameter
        m = MobileNet()
        print("Filename =", "test_"+str(i)+".pth")
        m.load_state_dict(torch.load("test_"+str(i)+".pth")['net'])
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

    torch.save(pushedModels[0], "optimizedModel.pth")


if __name__ == '__main__':
    logging.basicConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--compressFlag", help="Compression enabled/disabled")

    args = parser.parse_args()
    if args.compressFlag == "Y": 
        compressFlag = True
        
    print("Compression {} enabled".format(args.compressFlag))

    clients.append('localhost:50051')
    #clients.append('localhost:50052')

    run()
