
from concurrent import futures
import logging

import grpc
import federated_pb2
import federated_pb2_grpc
import main
import base64


class Trainer(federated_pb2_grpc.TrainerServicer):
    def StartTrain(self, request, context):
        main.train(1,request.rank, request.world)
       # print(main.net)
        f=open("checkpoint/ckpt.pth", "rb")
        encode=base64.b64encode(f.read())
       # print(encode)
        return federated_pb2.TrainReply(message=encode)
    def SendModel(self,request,context):
        f=open("checkpoint/ckpt.pth", "wb")
        decode=base64.b64decode(request.model)
        f.write(decode)
        f.close()
        return federated_pb2.SendModelReply(reply="success")



def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),options = [
        ('grpc.max_send_message_length', 1024 * 1024 * 1024),
        ('grpc.max_receive_message_length', 1024 * 1024 * 1024)
    ])
    federated_pb2_grpc.add_TrainerServicer_to_server(Trainer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()
