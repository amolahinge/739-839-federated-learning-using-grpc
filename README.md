[This code's documentation lives on the grpc.io site.](https://grpc.io/docs/languages/python/quickstart)

python3 -m grpc_tools.protoc -I../../739-839-federated-learning-using-grpc --python_out=. --grpc_python_out=. ../../739-839-federated-learning-using-grpc/federated.proto


For enabling compression, run both client and server with -c Y
`python3 client.py -c Y
python3 server.py -c Y`


For passing address to client, run client with -a addresss:port
`python3 client.py -a localhost:50051
python3 client.py -a localhost:50052`