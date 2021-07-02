import logging
from ..modules import Server

def start_server():
	logging.basicConfig(filename="mylog.log")
	logging.root.setLevel(logging.DEBUG)
	s = Server()
	s.server_loop()