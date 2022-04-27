from __future__ import print_function

import logging

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
optimModelPath = "optimizedModel.pth"

def trainThreadFunc(count,channel):
    net = MobileNet()
    stub = federated_pb2_grpc.TrainerStub(channel)
    response = stub.StartTrain(federated_pb2.TrainRequest(name='you',rank=count,world=len(clients)))
    model=base64.b64decode(response.message)
    f = open("test_"+str(count)+".pth",'wb')
    f.write(model)
    f.close()

def sendThreadFunc(count,channel):
    
    f=open(optimModelPath, "rb")
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
        for count,channel in enumerate(channels):
            sendThreads.append(threading.Thread(target=sendThreadFunc, args=(count,channel)))
        for i in range(len(sendThreads)):
            sendThreads[i].start()
        for i in range (len(sendThreads)):
            sendThreads[i].join()

def allreduce():
    pushedModels = []
    for i in range(0, len(clients)):
        m = MobileNet()
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

    state = {
        'net': optimizedModel,
        'acc': 1,
        'epoch': 1,
    }
    torch.save(state, optimModelPath)


if __name__ == '__main__':
    logging.basicConfig()
    clients.append('localhost:50051')
    #clients.append('localhost:50052')

    run()
