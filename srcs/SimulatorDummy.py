import signal
import sys
from simlaunch3000 import Client

import gym
import uuid
# import gym_donkeycar ## Keep this module 
import sys


class SimulatorDummy:
	def __init__(self, env_name):
		exe_path = "manual"
		self.client = Client()
		self.client.request_simulator()
		self.port = self.client.sim_port
		# port = 9093
		conf = {"exe_path": exe_path,
						"host": "127.0.0.1",
						"port": self.port,
						"body_style": "donkey",
						"body_rgb": (128, 128, 128),
						"car_name": "me",
						"font_size": 100,
						"racer_name": "DDQN",
						"country": "FR",
						"bio": "Learning to drive w DDQN RL",
						"guid": str(uuid.uuid4()),
						"max_cte": 10,
				}
		self.env = gym.make(env_name, conf=conf)

		env = self.env

		def signal_handler(signal, frame):
			print("catching ctrl+c")
			env.unwrapped.close()
			sys.exit(0)

		signal.signal(signal.SIGINT, signal_handler)
		signal.signal(signal.SIGTERM, signal_handler)
		signal.signal(signal.SIGABRT, signal_handler)
