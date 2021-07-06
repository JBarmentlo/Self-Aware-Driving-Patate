import logging
logging.basicConfig(filename="mylog.log")
logging.root.setLevel(logging.DEBUG)
from .utils import start_server
from .modules import Client, Server