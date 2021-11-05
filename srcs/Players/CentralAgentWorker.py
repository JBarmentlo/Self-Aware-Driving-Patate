from numpy.core.fromnumeric import mean
from ModelCache import ModelCache
import torch
import logging
import time
import numpy as np
import json
import io
from collections import deque

from RewardOpti import Logger, RewardOpti
from agents.Agent import DQNAgent
from Preprocessing import PreprocessingAE, PreprocessingVannilla
from S3 import S3
from utils import fix_cte, is_stuck
import utils
from SimCache import SimCache
import torch.distributed.rpc as rpc

from Simulator import Simulator
from config import DotDict
from Scorer import DistScorer




class CentralAgentWorker():
	agent: 		DQNAgent
	simulator: 	Simulator
	RO:			RewardOpti

	def __init__(self, config, rank, env_name = "donkey-circuit-launch-track-v0"):
		self.id = rpc.get_worker_info().id
		self.config = config.config_NeuralPlayer
		self.preprocessor = None
		self.rank = rank # * Starts at 1 for first worker
		self.simulator = Simulator(config.config_Simulator, env_name)
		self.simulator = utils.fix_cte(self.simulator)
		self.env = self.simulator.env
		self.S3 = S3(self.config.config_Datasets.S3_bucket_name)
		self._init_preprocessor(self.config.config_Preprocessing)
		self._init_reward_optimizer(self.config)
		self._init_logger()
		self.e = 0
		self.Logger = logging.getLogger(f"Central Agent Worker {self.rank}")
		self.Logger.setLevel(logging.WARNING)
		stream = logging.StreamHandler()
		self.Logger.addHandler(stream)

		# self._save_config()


	def _init_dataset(self, config):
		self.S3 = None
		if self.config.config_Datasets.S3_connection == True:
			self.S3 = S3(self.config.config_Datasets.S3_bucket_name)
		if self.config.agent_name == "DQN":
			self.SimCache = SimCache(self.config.config_Datasets.ddqn.sim, self.S3)


	def _init_preprocessor(self, config_Preprocessing):
		if (config_Preprocessing.use_AutoEncoder):
			self.preprocessor = PreprocessingAE(config = config_Preprocessing, S3=self.S3)
		else:
			self.preprocessor = PreprocessingVannilla(config = config_Preprocessing, S3=self.S3)


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
			Logger.warn("\nDone\n")
			return True #TODO : MAYBE COMMENTING THIS BREAKS SOMETHING

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


	def non_blocking_fix_cte(self, simulator, agent_rref, n_max):
		'''
			Returns a simulator instance containing a functional env
		'''
		cte = 100
		while(abs(cte) > 10 and not agent_rref.rpc_sync().is_enough_frames_generated(n_max)):
			state = simulator.env.reset()
			new_state, reward, done, infos = simulator.env.step([0, 1])

			if (abs(infos["cte"]) > 10):
				self.Logger.info(f"Attempting to fix broken cte by driving forward a little bit. cte: {infos['cte']}")
				new_state, reward, done, infos = simulator.env.step([0, 1])
				time.sleep(0.5)
				self.Logger.info(f"One step more. cte: {infos['cte']}")
			if (abs(infos["cte"]) > 10):
				new_state, reward, done, infos = simulator.env.step([0.1, 1])
				time.sleep(0.5)
				self.Logger.info(f"One step more. cte: {infos['cte']}")
			if (abs(infos["cte"]) > 10):
				new_state, reward, done, infos = simulator.env.step([-0.1, 1])
				time.sleep(1)
				self.Logger.info(f"One step more. cte: {infos['cte']}")
			if (abs(infos["cte"]) > 10):
				new_state, reward, done, infos = simulator.env.step([0, 1])
				time.sleep(0)
				self.Logger.info(f"One step more. cte: {infos['cte']}")
			
			cte = infos["cte"]
			if (abs(cte) > 10):
				self.Logger.warning(f"restarting sim because cte is fucked {cte}")
				simulator.restart_simulator()
		
		# simulator.env.reset()
		return simulator


	def do_races(self, agent_rref, n_max):
		self.agent_rref = agent_rref
		n = 0
		Scorer = None
		while (not self.agent_rref.rpc_sync().is_enough_frames_generated(n_max)):
			self.RO.new_race_init(self.e)
			self.e += 1
			self.simulator = fix_cte(self.simulator)
			self.env = self.simulator.env
			state, reward, done, infos = self.env.step([0, 0.1])
			processed_state = self.preprocessor.process(state)
			done = self._is_over_race(infos, done)
			self.Logger.debug(f"Initial CTE: {infos['cte']}")
			total_frames = 0
			Scorer = DistScorer()
			Scorer.first_point(infos)
			dists = deque(maxlen = 30)
			last_dist  = 0
			dist_diff = 0
			while (not (done or is_stuck(dists, 0.05))):
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
				if (n % 64 == 0):
					total_frames = self.agent_rref.rpc_sync().total_frames_generated()
				Scorer.add_point(infos)
				dist_diff = Scorer.get_current_race_dist() - last_dist
				last_dist = Scorer.get_current_race_dist()
				dists.append(dist_diff)
			Scorer.end_race(n)

		self.env.reset()
		if (Scorer == None):
			return([],[],[])
		return (Scorer.scores, Scorer.distances, Scorer.speeds)


	def do_eval_races(self, agent_rref, max_frames = 5000):
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
		Scorer = DistScorer()
		Scorer.first_point(infos)
		dists = deque(maxlen = 30)
		last_dist  = 0
		dist_diff = 0
		while (not (done or is_stuck(dists, 0.05))):
			action = self.get_action(processed_state)
			self.Logger.debug(f"action: {action}")
			new_state, reward, done, infos = self.env.step(action)
			new_processed_state = self.preprocessor.process(new_state)
			done = self._is_over_race(infos, done)
			# reward = self.RO.sticks_and_carrots(action, infos, done)
			# [action, reward] = utils.to_numpy_32([action, reward])
			self.agent_rref.rpc_async().increase_frame_count()
			processed_state = new_processed_state
			self.Logger.debug(f"cte:{infos['cte'] + 2.25}")
			Scorer.add_point(infos)

			dist_diff = Scorer.get_current_race_dist() - last_dist
			last_dist = Scorer.get_current_race_dist()
			dists.append(dist_diff)
			n = n + 1
		
		Scorer.end_race(n)
		# return (Scorer.scores, Scorer.distances, Scorer.speeds)
		return (Scorer.get_current_race_score(), Scorer.get_current_race_dist(), Scorer.get_current_race_speed())
