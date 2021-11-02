from inputs import get_key
import threading
import torch
import logging

from RewardOpti import RewardOpti
from agents.Agent import DQNAgent
from Preprocessing import Preprocessing
from Simulator import Simulator
# from srcs import Scorer
import utils
from S3 import S3
from SimCache import SimCache
from pynput import keyboard
from Scorer import DistScorer

Logger = logging.getLogger("HumanPlayer")
Logger.setLevel(logging.INFO)
stream = logging.StreamHandler()
Logger.addHandler(stream)


class GetKey():
	def __init__(self):
		self.on_press = False
		self.on_release = False
		self.key = None
		self.released_key = None
		
	def pressing(self, key):
		self.on_press = True
		self.on_release = False
		self.released_key = None
		try:
			self.key = key.char
		except AttributeError:
			self.key = key
		

	def releasing(self, key):
		self.on_press = False
		self.on_release = True
		self.key = None
		self.released_key = key
		if key == keyboard.Key.esc:
			return False




class HumanPlayer():
	def __init__(self, config, env, simulator):
		self.config = config
		self.env = env
		self.simulator = simulator
		self.throttle, self.steering = 0, 0
		self._init_dataset(config.config_Datasets)
		Logger.info("Human Player created")


	def _init_dataset(self, config):
		self.S3 = None
		if self.config.config_Datasets.S3_connection == True:
			self.S3 = S3(self.config.config_Datasets.S3_bucket_name)
		Logger.info("Initializing simcache")
		self.SimCache = SimCache(self.config.config_Datasets.sim, self.S3)
	
	
	def add_simcache_point(self, datapoint):
		self.SimCache.add_point(datapoint)
		

	def get_action(self, key, released_key):
		if key != None and type(key) != str:
			if key == key.up:
				self.throttle = min(self.config.max_throttle, max(self.config.min_throttle, abs(self.throttle * self.config.coef)))
			elif key == key.down:
				self.throttle = min(self.config.min_throttle, max(self.config.min_throttle, abs(self.throttle * self.config.coef) * 1))
				self.throttle = -1.0

			elif key == key.left:
				if self.steering == 0:
					self.steering = min(self.config.max_steering, max(self.config.min_steering, self.config.init_steering * -1))
				else:
					self.steering = min(self.config.max_steering, max(self.config.min_steering, abs(self.steering * self.config.coef) * -1))
			elif key == key.right:
				if self.steering == 0:
					self.steering = min(self.config.max_steering, max(self.config.min_steering, self.config.init_steering))
				else:
					self.steering = min(self.config.max_steering, max(self.config.min_steering, abs(self.steering * self.config.coef)))
		elif released_key != None:
			if released_key == released_key.up or released_key == released_key.down:
				self.throttle = self.config.init_throttle
			elif released_key == released_key.left or released_key == released_key.right:
				self.steering = 0
		
		self.check_min_max()
		return ([self.steering, self.throttle])
	
	def check_min_max(self):
		if self.throttle > self.config.max_throttle:
			self.throttle = self.config.max_throttle
		elif self.throttle < self.config.min_throttle:
			self.throttle = self.config.min_throttle
			
		
	
	def do_race(self):
		Logger.info(f"Starting Human race:")
		self.simulator = utils.fix_cte(self.simulator)
		self.env = self.simulator.env
		action = [0, 0]
		state, reward, done, infos = self.env.step(action)

		gk = GetKey()
		listener =  keyboard.Listener(
				on_press=gk.pressing,
			on_release=gk.releasing)
		print("listening Keyboard: you can start driving\n")
		listener.start()
		scor = DistScorer()
		scor.first_point(infos)
		while listener.running:
			action = self.get_action(gk.key, gk.released_key)
			Logger.debug(f"action: {action}")
			new_state, reward, done, infos = self.env.step(action)
			self.add_simcache_point([state, action, new_state, reward, done, infos])
			state = new_state
			scor.add_point(infos)
			print(scor.current_race_dist)

		listener.stop()
		print("\nStoping race and listening Keyboard\n")

		self.SimCache.upload(f"{self.config.config_Datasets.sim.save_name}")
		self.env.reset()
		return
