import sys
from simlaunch3000 import Client

import gym
import uuid
import gym_donkeycar ## Keep this module 
import sys
import time


class Simulator:
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
		# Details: gym.make() do not wait enough time for the simulator to start
		# time.sleep(3)
		try:
			self.env = gym.make(self.env_name, conf=self.config)
		except:
			time.sleep(4)
			self.env = gym.make(self.env_name, conf=self.config)
	
	def new_track(self):
		print("Making new track")
		self.env.viewer.exit_scene()
		# self.env.close()
		self.env = gym.make(self.env_name, conf=self.config)

	def release_simulator(self):
		self.client.release_sim(self.port)


	def kill_simulator(self):
		self.env.close()
		self.client.kill_sim()


	def restart_simulator(self):
		self.kill_simulator()
		self.start_simulator()