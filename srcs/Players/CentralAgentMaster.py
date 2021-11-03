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

Logger = logging.getLogger("Central Agent Master")
Logger.setLevel(logging.INFO)
stream = logging.StreamHandler()
Logger.addHandler(stream)


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
		self.scores = []
		self.agent_rref = RRef(self.agent)
		self.world_size = world_size #nb of remote agents
		self.worker_rrefs = []
		self.scores_tmp = []
		self.scores = []
		for worker_rank in range(1, self.world_size):
			worker_info = rpc.get_worker_info(f"worker{worker_rank}")
			self.worker_rrefs.append(remote(worker_info, CentralAgentWorker, args = (config, ), timeout=600))
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


	def run_remote_episode(self, num_frames = 10):
		self.agent.new_frames = 0
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
			
		for _ in range(self.config.replay_memory_batches):
			self.agent.replay_memory()

		self.agent._update_epsilon()

		if (self.agent.config.data.saving_frequency != 0 and (self.e % self.agent.config.data.saving_frequency == 0 or self.e == self.config.episodes)):
			self.agent.ModelCache.save(self.agent.model, f"{self.agent.config.data.save_name}{self.e}")

		self.e += 1


	def run_eval_episode(self):
		print("EVAL EPISODE\n")
		self.scores_tmp = []
		tmp_epsilone = self.agent.config.epsilon
		self.agent.config.epsilon = 0.0
		futures = []
		for worker_rref in self.worker_rrefs:
			futures.append(
				rpc_async(
					worker_rref.owner(),
					worker_rref.rpc_sync(timeout=0).do_eval_races,
					args=(self.agent_rref,),
					timeout=0
				)
			)

		for fut in futures:
			fut.wait()

		self.agent.config.epsilon = tmp_epsilone

		for fut in futures:
			scor = fut.value()
			self.scores_tmp.append(scor)
			print(f"recieved score {scor}")

		self.scores.append(np.mean(self.scores_tmp))
		self.scores_tmp = []
		print(f"Mean score {self.scores[-1]}\n\n")


	def save(self, name):
		self.agent.save(name)