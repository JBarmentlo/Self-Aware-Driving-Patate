from torch.distributed.rpc.api import rpc_sync
from ModelCache import ModelCache
import torch
import logging
import time
import numpy as np
import json
import io

from RewardOpti import RewardOpti
from agents.Agent import DQNAgent
from S3 import S3
import utils
from SimCache import SimCache
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef, rpc_async, remote
from .CentralAgentWorker import CentralAgentWorker
from Simulator import Simulator
from config import DotDict

from datetime import datetime
from typing import List
import pandas as pd


Logger = logging.getLogger("Central Agent Master")
Logger.setLevel(logging.INFO)
stream = logging.StreamHandler()
Logger.addHandler(stream)

import matplotlib
# matplotlib.use('TkAgg')dist_diff
import matplotlib.pyplot as plt

class ScoreDataGatherer():
	'''
		Container for the scores, speeds and distances of the races,
	'''
	def __init__(self):
		self.data		: pd.DataFrame = pd.DataFrame(columns=["score", "speed", "dist", "episode", "epsilon", "worker_id"])
		self.eval_data	: pd.DataFrame = pd.DataFrame(columns=["score", "speed", "dist", "episode", "epsilon", "worker_id"])
		# plt.ion()


	def add_point(self, score, speed, dist, episode, epsilon, worker_id):
		self.data = self.data.append({"score" : score, "speed" : speed, "dist" : dist, "episode" : episode, "epsilon" : epsilon, "worker_id" : worker_id}, ignore_index= True)


	def add_eval_point(self, score, speed, dist, episode, epsilon, worker_id):
		self.eval_data = self.eval_data.append({"score" : score, "speed" : speed, "dist" : dist, "episode" : episode, "epsilon" : epsilon, "worker_id" : worker_id}, ignore_index= True)


	def plot(self):
		'''
			will hang until closed
		'''
		ax = self.data.plot(kind = "scatter", x = "episode", y = "score", color = "C3", label = "Score")
		# self.data.plot(kind = "scatter", x = "episode", y = "speed", color = "C2", label = "Speed", ax = ax)
		# self.data.plot(kind = "scatter", x = "episode", y = "dist", color = "C4", label = "Distance", ax = ax)
		# plt.draw()
		# plt.pause(0.1)
		print("DATAAA\n\n\n")
		print(self.data)
		# plt.savefig("./CURRENT_SCORE")



class CentralAgentMaster():
	agent: 		DQNAgent
	simulator: 	Simulator
	RO:			RewardOpti

	def __init__(self, config, world_size):
		self.e = 0
		self.config = config.config_NeuralPlayer
		self.preprocessor = None
		self._init_dataset(self.config.config_Datasets)
		self._init_agent(self.config.config_Agent)
		self.agent_rref = RRef(self.agent)
		self.world_size = world_size #nb of remote agents
		self.worker_rrefs = []
		self.data_gatherer = ScoreDataGatherer()
		for worker_rank in range(1, self.world_size):
			worker_info = rpc.get_worker_info(f"worker{worker_rank}")
			self.worker_rrefs.append(remote(worker_info, CentralAgentWorker, args = (config, worker_rank), timeout=600))
		# self._save_config()


	def _init_dataset(self, config):
		self.S3 = None
		if self.config.config_Datasets.S3_connection == True:
			self.S3 = S3(self.config.config_Datasets.S3_bucket_name)
		if self.config.agent_name == "DQN":
			self.SimCache = SimCache(self.config.config_Datasets.ddqn.sim, self.S3)


	def _init_agent(self, config_Agent):
		self.agent = DQNAgent(config=config_Agent, S3=self.S3)
  
  
	def _init_reward_optimizer(self, config_NeuralPlayer):
		self.RO = RewardOpti(config_NeuralPlayer)


	def update_worker_agent_params(self):
		state_dict = {k: v.cpu() for k, v in zip(self.agent.model.state_dict().keys(), self.agent.model.state_dict().values())}
		futures = []
		for worker_rref in self.worker_rrefs:
			futures.append(
				rpc_async(
					worker_rref.owner(),
					worker_rref.rpc_sync(timeout=0).update_agent_params,
					args=(state_dict,),
					timeout=0
				)
			)

		for fut in futures:
			fut.wait()


	def run_remote_episode(self, num_frames, episode_num):
		self.agent.new_frames = 0
		self.agent.all_workers_done = [False] * self.world_size
		self.agent.create_loading_bar(num_frames)
		futures = []
		for worker_rref in self.worker_rrefs:
			futures.append(
				rpc_async(
					worker_rref.owner(),
					worker_rref.rpc_sync(timeout=0).do_races,
					args=(self.agent_rref, num_frames),
					timeout=0
				)
			)
		
		for fut in futures:
			fut.wait()

		for id, fut in enumerate(futures):
			scores, distances, speeds = fut.value()
			for s, d, sp in zip(scores, distances, speeds):
				# print("adding: ", s, sp, d, episode_num, self.agent.config.epsilon)
				self.data_gatherer.add_point(s, sp, d, episode_num, self.agent.config.epsilon, id)


		for _ in range(self.config.replay_memory_batches):
			Logger.debug(f"Replay from memory")
			self.agent.replay_memory()

		self.agent._update_epsilon()

		if (self.agent.config.data.saving_frequency != 0 and (self.e % self.agent.config.data.saving_frequency == 0 or self.e == self.config.episodes)):
			self.agent.ModelCache.save(self.agent.model, f"{self.agent.config.data.save_name}{self.e}")

		self.e += 1


	def run_eval_episode(self, episode_num, max_frames):
		print("EVAL EPISODE\n")
		self.scores_tmp = []
		tmp_epsilone = self.agent.config.epsilon
		self.agent.config.epsilon = 0.0
		futures = []
		self.agent.create_loading_bar(max_frames)
		for worker_rref in self.worker_rrefs:
			futures.append(
				rpc_async(
					worker_rref.owner(),
					worker_rref.rpc_sync(timeout=0).do_eval_races,
					args=(self.agent_rref, max_frames),
					timeout=0
				)
			)

		for fut in futures:
			fut.wait()

		for id, fut in enumerate(futures):
			score, distance, speed = fut.value()
			self.data_gatherer.add_point(score, speed, distance, episode_num, self.agent.config.epsilon, id)

		self.agent.config.epsilon = tmp_epsilone


	def save(self, name):
		self.agent.save(name)