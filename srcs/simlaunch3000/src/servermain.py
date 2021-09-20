import logging
from modules import PortHandler
from modules import Server

def start_server():
	logging.basicConfig(filename="mylog.log")
	logging.root.setLevel(logging.DEBUG)
	s = Server()
	s.server_loop()

if __name__ == "__main__":
	start_server()