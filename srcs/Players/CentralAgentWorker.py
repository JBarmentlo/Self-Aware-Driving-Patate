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
from Score import DistanceTracker

from Simulator import Simulator
from config import DotDict
from Scorer import DistScorer

class CentralAgentWorker():
	agent: 		DQNAgent
	simulator: 	Simulator
	RO:			RewardOpti

	def __init__(self, config, env_name = "donkey-generated-roads-v0"):
		self.id = rpc.get_worker_info().id
		self.config = config.config_NeuralPlayer
		self.preprocessor = None
		self.simulator = Simulator(config.config_Simulator, env_name)  
		self.simulator = utils.fix_cte(self.simulator)
		self.env = self.simulator.env
		self.S3 = S3(self.config.config_Datasets.S3_bucket_name)
		self._init_preprocessor(self.config.config_Preprocessing)
		self._init_reward_optimizer(self.config)
		self._init_logger()
		self.e = 0
		# self._save_config()


	def _init_dataset(self, config):
		self.S3 = None
		if self.config.config_Datasets.S3_connection == True:
			self.S3 = S3(self.config.config_Datasets.S3_bucket_name)
		if self.config.agent_name == "DQN":
			self.SimCache = SimCache(self.config.config_Datasets.ddqn.sim, self.S3)


	def _init_preprocessor(self, config_Preprocessing):
		self.preprocessor = Preprocessing(config = config_Preprocessing, S3=self.S3)


	def update_agent_params(self, state_dict):
		pass


	def _init_logger(self):
		self.Logger = logging.getLogger(f"DistributedPlayer-{self.id}")
		self.Logger.setLevel(logging.INFO)
		self.Logger.addHandler(logging.StreamHandler())


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
		while (not self.agent_rref.rpc_sync().is_enough_frames_generated(n_max)):
			self.RO.new_race_init(self.e)
			self.e += 1
			# self.simulator.new_track()  destroys cte.
			self.simulator = utils.fix_cte(self.simulator)
			self.env = self.simulator.env

			state, reward, done, infos = self.env.step([0, 0.1])
			processed_state = self.preprocessor.process(state)
			done = self._is_over_race(infos, done)
			self.Logger.debug(f"Initial CTE: {infos['cte']}")
			while (not done):
				action = self.get_action(processed_state)
				self.Logger.debug(f"action: {action}")
				new_state, reward, done, infos = self.env.step(action)
				new_processed_state = self.preprocessor.process(new_state)
				done = self._is_over_race(infos, done)
				reward = self.RO.sticks_and_carrots(action, infos, done)
				[action, reward] = utils.to_numpy_32([action, reward])
				self.agent_rref.rpc_async().add_to_memory(processed_state, action, new_processed_state, reward, done)
				processed_state = new_processed_state
				self.Logger.debug(f"cte:{infos['cte'] + 2.25}")
				n += 1

		return


	def do_eval_races(self, agent_rref):
		self.agent_rref = agent_rref
		n = 0
		self.RO.new_race_init(self.e)
			# self.simulator.new_track()  destroys cte.
		self.simulator = utils.fix_cte(self.simulator)
		self.env = self.simulator.env

		state, reward, done, infos = self.env.step([0, 0.1])
		processed_state = self.preprocessor.process(state)
		done = self._is_over_race(infos, done)
		self.Logger.debug(f"Initial CTE: {infos['cte']}")
		# Scorer = DistanceTracker(infos["pos"], infos["cte"])
		Scorer = DistScorer()
		Scorer.first_point(infos)
		while (not done):
			action = self.get_action(processed_state)
			self.Logger.debug(f"action: {action}")
			new_state, reward, done, infos = self.env.step(action)
			new_processed_state = self.preprocessor.process(new_state)
			done = self._is_over_race(infos, done)
			reward = self.RO.sticks_and_carrots(action, infos, done)
			[action, reward] = utils.to_numpy_32([action, reward])
			# self.agent_rref.rpc_async().add_to_memory(processed_state, action, new_processed_state, reward, done)
			processed_state = new_processed_state
			self.Logger.debug(f"cte:{infos['cte'] + 2.25}")
			Scorer.add_point(infos)
		
		
		return Scorer.current_race_dist
