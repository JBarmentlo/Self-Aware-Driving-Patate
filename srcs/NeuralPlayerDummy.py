from agents.AgentDummy import DQNAgent
from PreprocessingDummy import Preprocessing
import torch
import logging
import time
import numpy as np

Logger = logging.getLogger("NeuralPlayer")
Logger.setLevel(logging.INFO)
stream = logging.StreamHandler()
Logger.addHandler(stream)


class NeuralPlayerDummy():
	def __init__(self, config, env, simulator):
		self.config = config
		self.env = env
		self.agent =  None
		self.preprocessor = None
		self.simulator = simulator
		self._init_agent(config.config_Agent)
		self._init_preprocessor(config.config_Preprocessing)



	def _init_preprocessor(self, config_Preprocessing):
		self.preprocessor = Preprocessing(config = config_Preprocessing)


	def _init_agent(self, config_Agent):
		self.agent = DQNAgent(config = config_Agent)


	def _train_agent(self):
		self.agent.train()


	def _is_over_race(self, info, done):
		cte = info["cte"]
		cte_corr = cte + self.config.cte_offset
		if (done):
			return True

		if (abs(cte) > 100):
			return True
		
		if (abs(cte_corr) > self.config.cte_limit):
			return True

		return False


	def get_action(self, state):
		return self.agent.get_action(self.preprocessor.process(state))


	def do_races(self, episodes = None):
		for e in range(1, episodes + 1):
			cte = 100

			while(abs(cte) > 1):
				state = self.env.reset()
				new_state, reward, done, info = self.env.step([0, 1])

				if (abs(info["cte"]) > 1):
					Logger.warn(f"Attempting to fix broken cte by driving forward a little bit. cte: {info['cte']}")
					new_state, reward, done, info = self.env.step([0, 1])
					time.sleep(0.5)
					Logger.warn(f"One step more. cte: {info['cte']}")
				if (abs(info["cte"]) > 1):
					new_state, reward, done, info = self.env.step([0.1, 1])
					time.sleep(0.5)
					Logger.warn(f"One step more. cte: {info['cte']}")
				if (abs(info["cte"]) > 1):
					new_state, reward, done, info = self.env.step([-0.1, 1])
					time.sleep(1)
					Logger.warn(f"One step more. cte: {info['cte']}")
				if (abs(info["cte"]) > 1):
					new_state, reward, done, info = self.env.step([0, 1])
					time.sleep(0)
					Logger.warn(f"One step more. cte: {info['cte']}")
				
				cte = info["cte"]
				if (abs(cte) > 1):
					Logger.warning(f"restarting sim because cte is fucked {cte}")
					self.simulator.restart_simulator()
					self.env = self.simulator.env


			processed_state = self.preprocessor.process(state)
			done = self._is_over_race(info, done)
			Logger.debug(f"Initial CTE: {info['cte']}")
			done = False
			while (not done):
				action = self.agent.get_action(processed_state, e)
				Logger.debug(f"action: {action}")
				# steering, throttle = action[0], action[1]
				old_info = info
				new_state, reward, done, info = self.env.step(action)
				new_processed_state = self.preprocessor.process(new_state)
				done = self._is_over_race(info, done)
				self.agent.memory.add(processed_state, action, new_processed_state, reward, done)
				processed_state = new_processed_state
				Logger.debug("cte:", info["cte"] + 2.25)

			self.agent._update_epsilon()
			self.agent.replay_memory()
			# if (e % self.config.train_frequency == 0):
			# 	self.train_agent()
		self.env.reset()
		return
