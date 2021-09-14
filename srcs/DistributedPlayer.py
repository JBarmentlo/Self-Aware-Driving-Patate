from ModelCache import ModelCache
import torch
import logging
import time
import numpy as np
import json
import io

from RewardOpti import RewardOpti
from agents.Agent import DQNAgent
from Preprocessing import Preprocessing
from S3 import S3
import utils
from SimCache import SimCache
import torch.distributed.rpc as rpc

Logger = logging.getLogger("NeuralPlayer")
Logger.setLevel(logging.INFO)
stream = logging.StreamHandler()
Logger.addHandler(stream)

from Simulator import Simulator
from config import DotDict

class DistributedPlayer():
	def __init__(self, config, env_name = "donkey-generated-roads-v0"):
		# config = DotDict(config)
		self.config = config.config_NeuralPlayer
		self.agent =  None
		self.preprocessor = None
		self.simulator = Simulator(config.config_Simulator, env_name)  
		self.simulator = utils.fix_cte(self.simulator)
		self.env = self.simulator.env
		self._init_preprocessor(self.config.config_Preprocessing)
		self._init_reward_optimizer(self.config)
		self.scores = []
		self.id = rpc.get_worker_info().id
		# self._save_config()


	def _init_preprocessor(self, config_Preprocessing):
		self.preprocessor = Preprocessing(config = config_Preprocessing)


	def _init_reward_optimizer(self, config_NeuralPlayer):
		self.RO = RewardOpti(config_NeuralPlayer)


	def _is_over_race(self, infos, done):
		cte = infos["cte"]
		cte_corr = cte + self.config.cte_offset
		if (done):
			return True

		if (abs(cte) > 100):
			return True
		
		if (abs(cte_corr) > self.config.cte_limit):
			return True

		return False



	def get_action(self, processed_state):
		return self.agent_rref.rpc_sync().get_action(processed_state)


	def release_sim(self):
		self.env.close()
		self.simulator.release_simulator()


	def do_races(self, agent_rref, n_max):
		self.agent_rref = agent_rref
		n = 0
		e = 0
		while (n < n_max):
			self.RO.new_race_init(e)
			e += 1
			self.simulator = utils.fix_cte(self.simulator)
			self.env = self.simulator.env

			state, reward, done, infos = self.env.step([0, 0])
			processed_state = self.preprocessor.process(state)
			done = self._is_over_race(infos, done)
			Logger.debug(f"Initial CTE: {infos['cte']}")
			while (not done):

				action = self.get_action(processed_state)
				Logger.debug(f"action: {action}")
				new_state, reward, done, infos = self.env.step(action)
				new_processed_state = self.preprocessor.process(new_state)
				done = self._is_over_race(infos, done)
				reward = self.RO.sticks_and_carrots(action, infos, done)
				[action, reward] = utils.to_numpy_32([action, reward])
				self.agent_rref.rpc_sync().add_to_memory(processed_state, action, new_processed_state, reward, done)
				processed_state = new_processed_state
				Logger.debug(f"cte:{infos['cte'] + 2.25}")
				n += 1
				print(f"{n = }")
				if (n > n_max):
					break

			if (n > n_max):
				break

		self.env.reset()
		return
