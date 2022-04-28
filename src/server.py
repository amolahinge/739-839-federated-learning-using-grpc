from __future__ import print_function

import logging
import argparse
import time
import copy

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


channels={} #client - client channel
clients={} # client - active / inactive status
compressFlag = False
optimModelPath = "optimizedModel.pth"

options = [
    ('grpc.max_send_message_length', 1024 * 1024 * 1024),
    ('grpc.max_receive_message_length', 1024 * 1024 * 1024)
    ]

def trainThreadFunc(count, client, channel):
    stub = federated_pb2_grpc.TrainerStub(channel)
    try:
        response = stub.StartTrain(federated_pb2.TrainRequest(rank=count,world=len(clients)))
        model=base64.b64decode(response.message)
        f = open("test_"+str(count)+".pth",'wb')
        f.write(model)
        f.close()
    except grpc.RpcError as e:
        status_code = e.code()
        #if grpc.StatusCode.UNAVAILABLE == status_code:
        clients[client] = False

def sendThreadFunc(count,client, channel):
    
    f=open(optimModelPath, "rb")
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
                        sendThreadFunc(list(clients_copy.keys()).index(key),key,channel)
                except grpc.RpcError as e:
                    status_code = e.code()
                    #print("GRPC Error", status_code)
        print("Client status", clients)

def createChannel(client):
    if compressFlag:
        return grpc.insecure_channel(client,options=options,compression=grpc.Compression.Gzip)
    else:
        return grpc.insecure_channel(client,options=options)

def init():
    for client in clients.keys():
        channels[client] = createChannel(client)

def run():
    init()
    #fault tolerance
    clientTracking = threading.Thread(target=checkClientStatus)
    clientTracking.start()

    for epoch in range(20):
        print("Starting epoch", epoch)
        trainthreads=[]
        #print(len(channels))
        count = 0
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

        for client,channel in channels.items():
            if clients[client]:
                sendThreads.append(threading.Thread(target=sendThreadFunc, args=(count,client,channel)))
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
        m.load_state_dict(torch.load("test_"+str(i)+".pth")['net'])
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
    torch.save(state, optimModelPath)


if __name__ == '__main__':
    logging.basicConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--compressFlag", help="Compression enabled/disabled")

    args = parser.parse_args()
    if args.compressFlag == "Y": 
        compressFlag = True        
    print("Compression {} enabled".format(args.compressFlag))
    
    clients['localhost:50051'] = True
    clients['localhost:50052'] = True
    #clients.append('localhost:50053')
    #clients.append('localhost:50054')
    run()
