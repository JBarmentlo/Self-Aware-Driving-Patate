import logging
logging.basicConfig(filename="mylog.log")
logging.root.setLevel(logging.DEBUG)
from .modules import Client, Server
from .start_server import start_server