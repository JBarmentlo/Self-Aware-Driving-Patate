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
		self.client.request_simulator()
		self.port = self.client.sim_port
		config.port = self.port
		self.env = gym.make(env_name, conf=config)

		env = self.env

		def signal_handler(signal, frame):
			print("catching ctrl+c")
			env.unwrapped.close()
			sys.exit(0)

		signal.signal(signal.SIGINT, signal_handler)
		signal.signal(signal.SIGTERM, signal_handler)
		signal.signal(signal.SIGABRT, signal_handler)
	