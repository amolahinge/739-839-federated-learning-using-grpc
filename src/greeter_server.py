
from concurrent import futures
import logging

import grpc
import helloworld_pb2
import helloworld_pb2_grpc
import main
import base64


class Greeter(helloworld_pb2_grpc.GreeterServicer):
    def SayHello(self, request, context):
        main.train(1,request.rank, request.world)
       # print(main.net)
        f=open("checkpoint/ckpt.pth", "rb")
        encode=base64.b64encode(f.read())
       # print(encode)
        return helloworld_pb2.HelloReply(message=encode)
    def SendModel(self,request,context):
        f=open("checkpoint/ckpt.pth", "wb")
        f.write(request.model)
        f.close()
        return helloworld_pb2.HelloReply(reply="success")



def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),options = [
        ('grpc.max_send_message_length', 1024 * 1024 * 1024),
        ('grpc.max_receive_message_length', 1024 * 1024 * 1024)
    ])
    helloworld_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()
