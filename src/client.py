
from concurrent import futures
import logging
import argparse

import grpc
import federated_pb2
import federated_pb2_grpc
import main
import base64

compressFlag = False
address = "temp"

class Trainer(federated_pb2_grpc.TrainerServicer):
    def StartTrain(self, request, context):
        print("Requesting train",datetime.now())

        main.train(1,request.rank, request.world)
        print("Train complete",datetime.now())
       # print(main.net)
        filePath = "./checkpoint/" + address + ".pth"
        f=open(filePath, "rb")
        encode=base64.b64encode(f.read())
        print("Encoding complete",datetime.now())

       # print(encode)
        return federated_pb2.TrainReply(message=encode)
    def SendModel(self,request,context):
        filePath = "./checkpoint/" + address + ".pth"
        f=open(filePath, "wb")
        decode=base64.b64decode(request.model)
        f.write(decode)
        f.close()
        main.test(1,1)
        return federated_pb2.SendModelReply(reply="success")

    def HeartBeat(self, request, context):
        #print("here")
        return federated_pb2.HeartBeatResponse(status=1)


def serve():
    if compressFlag:
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),options = [
            ('grpc.max_send_message_length', 1024 * 1024 * 1024),
            ('grpc.max_receive_message_length', 1024 * 1024 * 1024)
        ], compression=grpc.Compression.Gzip)
    else:
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),options = [
            ('grpc.max_send_message_length', 1024 * 1024 * 1024),
            ('grpc.max_receive_message_length', 1024 * 1024 * 1024)
        ])
    federated_pb2_grpc.add_TrainerServicer_to_server(Trainer(), server)
    server.add_insecure_port(address)
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--compressFlag", help="Compression enabled/disabled")
    parser.add_argument("-a", "--address", help="Listener port")
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

    args = parser.parse_args()
    if args.compressFlag == "Y": 
        compressFlag = True
    
    if args.address:
        address = args.address

    print("Client is running on {} ".format(address))
    print("Compression {} enabled".format(args.compressFlag))
    logging.basicConfig()
    serve()
