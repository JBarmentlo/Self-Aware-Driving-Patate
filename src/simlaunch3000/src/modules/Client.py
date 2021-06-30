from ..config import net_config
import socket
import json
import os
import logging

HOST = net_config.host  # The server's hostname or IP address
PORT = net_config.server_port        # The port used by the server

class  Client():
    def __init__(self):
        self.sim_port = None

    def request_simulator(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            msg = {"pass": os.environ["PS"], "req": net_config.start_sim_request} # a real dict.
            data = json.dumps(msg)
            try:
                # Connect to server and send data
                sock.connect((HOST, PORT))
                sock.sendall(bytes(data,encoding="utf-8"))
                received = sock.recv(1024)
                received = received.decode("utf-8")
                received = json.loads(received)
            except:
                pass
        try:
            print('Received', received)
            self.sim_port = received["sim_port"]
            return received
        except UnboundLocalError:
            print("Nothing recieved, is server started ?")
        except:
            self.sim_port = None
            return received

            




    def ping_sim(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            msg = {"pass": os.environ["PS"], "req": net_config.ping_request, "port" : self.sim_port} # a real dict.
            data = json.dumps(msg)
            try:
                # Connect to server and send data
                sock.connect((HOST, PORT))
                sock.sendall(bytes(data,encoding="utf-8"))
                received = sock.recv(1024)
                received = received.decode("utf-8")
            except:
                pass
        try:
            print('Received', received)
        except:
            pass


    def kill_sim(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            msg = {"pass": os.environ["PS"], "req": net_config.kill_request, "port" : self.sim_port} # a real dict.
            data = json.dumps(msg)
            try:
                # Connect to server and send data
                sock.connect((HOST, PORT))
                sock.sendall(bytes(data,encoding="utf-8"))
                received = sock.recv(1024)
                received = received.decode("utf-8")
            except:
                pass
        try:
            print('Received', received)
        except:
            pass

# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#     # s.sendall(int.to_bytes(9099, byteorder='little', length=1024))
#     m = {"port": 2020, "name": "abc"} # a real dict.


#     data = json.dumps(m)
#     # Create a socket (SOCK_STREAM means a TCP socket)
#     sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

#     try:
#         # Connect to server and send data
#         sock.connect((HOST, PORT))
#         sock.sendall(bytes(data,encoding="utf-8"))


#         # Receive data from the server and shut down
#         received = sock.recv(1024)
#         received = received.decode("utf-8")

#     finally:
#         data = s.recv(1024)

# print('Received', repr(data))