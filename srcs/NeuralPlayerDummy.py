from agents.AgentDummy import DQNAgent
from PreprocessingDummy import Preprocessing
import torch
import logging

Logger = logging.getLogger("NeuralPlayer")
Logger.setLevel(logging.WARN)
stream = logging.StreamHandler()
Logger.addHandler(stream)


class NeuralPlayerDummy():
	def __init__(self, config = None, env = None):
		self.config = config
		self.env = env
		self.agent =  None
		self.preprocessor = None
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
		for e in range(1, episodes):
			state = self.env.reset()
			processed_state = self.preprocessor.process(state)

			done = False
			while (not done):
				action = self.agent.get_action(processed_state, e)
				Logger.debug(f"action: {action}")
				# steering, throttle = action[0], action[1]
				new_state, reward, done, info = self.env.step(action)
				new_processed_state = self.preprocessor.process(new_state)
				done = self._is_over_race(info, done)
				self.agent.memory.add(torch.Tensor(processed_state), torch.Tensor(action), torch.Tensor(new_processed_state), reward, done)
				processed_state = new_processed_state
				print("cte     :", info["cte"])
				print("cte_corr:", info["cte"] + 2.25)

			# if (e % self.config.train_frequency == 0):
			# 	self.train_agent()
