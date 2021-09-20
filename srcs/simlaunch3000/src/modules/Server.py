import socket
import time
import json
import os
from . import SimHandler
from ..config import net_config
import logging

logging.root.setLevel(logging.DEBUG)
ServerLogger = logging.getLogger("Server")
ServerLogger.setLevel(logging.DEBUG)

class Server():
	'''
		Start a server with server.server_loop()
	'''
	def __init__(self):
		self.simhandler = SimHandler()
		ServerLogger.debug(f"\n\nStarted Server Instance")


	def server_loop(self):
		'''
			Main server loop. Listens to port, launches sims, answers port numbers of sims.
		'''
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
			try:
				ServerLogger.debug(f"Socket: {s} ")
				s.settimeout(10)
				s.bind((net_config.host, net_config.server_port))
				s.listen()
				ServerLogger.debug(f"Socket {s} bound, listening")
				while True:
					try:
						time.sleep(1)
						conn, addr = s.accept()
						with conn:
							ServerLogger.debug(f"Connected by {addr}")
							print(f"Connected by {addr}")
							data = conn.recv(1024)
							data = data.decode("utf-8")
							sim_port = None
							reply = {}
							try:
								data = json.loads(data)
								ServerLogger.debug(f"Msg received: {data}")
								print(f"Msg received: {data}")
								reply = self.handle_request(data)
								ServerLogger.debug(f"Reply: {reply}")
							except Exception as e:
								ServerLogger.error(f"{e}")
								print(f"error in main loop msg handle msg: {data} with error: {e}")
								pass
							conn.sendall(bytes(json.dumps(reply), encoding="utf-8"))
					except socket.timeout:
						pass
					self.simhandler.kill_idle_sims()
					self.simhandler.clean_dead_sims()
			except KeyboardInterrupt:
				s.close()
				conn.close()
				print("socket closed")
				ServerLogger.debug(f"Socket: {s} ")
				time.sleep(1)
				exit()


	def handle_request(self, data):
		if (data["pass"] != os.environ["PS"]):
			ServerLogger.error(f"Pass doesn match. Recieved: {data['pass']}")
			return {}

		if (data["req"] == net_config.start_sim_request):
			sim_port = self.simhandler.get_sim()
			return {"sim_port" : sim_port}

		if (data["req"] == net_config.ping_request):
			self.simhandler.ping_sim(data["port"])
			return {"pinged_sim" : data["port"]}

		if (data["req"] == net_config.kill_request):
			self.simhandler.kill_sim(data["port"])
			return {"killed_sim" : data["port"]}

		if (data["req"] == net_config.stop_using_sim_request):
			self.simhandler.release_sim(data["port"])
			return {"released_sim" : data["port"]}

		ServerLogger.error(f"Unhandled message: {data}")
		print("NOTHING sim")
		
