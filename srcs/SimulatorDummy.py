import signal
import sys
from simlaunch3000 import Client

import gym
import uuid
# import gym_donkeycar ## Keep this module 
import sys


class SimulatorDummy:
	def __init__(self, config, env_name):
		exe_path = "manual"
		self.client = Client()
		self.config = config
		self.env_name = env_name
		self.start_simulator()

		env = self.env

	def start_simulator(self):
		self.client.request_simulator()
		self.port = self.client.sim_port
		self.config.port = self.port
		# Dirty fix for Exception from gym.make()
		# Exception: Could not connect to server. Is it running? If you specified 'remote', then you must start it manually.
		import time
		time.sleep(2)
		# Details: gym.make() do not wait enough time for the simulator to start
		self.env = gym.make(self.env_name, conf=self.config)


	def kill_simulator(self):
		self.env.close()
		self.client.kill_sim()


	def restart_simulator(self):
		self.kill_simulator()
		self.start_simulator()

		
	def signal_handler(self, signal, frame):
		print("catching ctrl+c")
		# self.env.unwrapped.close()
		self.kill_simulator()
		sys.exit(0)

	# signal.signal(signal.SIGINT, signal_handler)
	# signal.signal(signal.SIGTERM, signal_handler)
	# signal.signal(signal.SIGABRT, signal_handler)
	