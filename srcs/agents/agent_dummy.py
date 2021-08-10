import numpy as np

class AgentDummy():
	def __init__(self, config_Agent = None):
		self.config = config_Agent
		self.memory = self._init_memory(config_Agent.config_Memory)
	

	def _init_memory(self, memory_config = None):
		pass


	def get_action(self, state, episode = 0):
		return (np.random.random((2)) - [0.5, 0]) * [6, 1]

	
	def train(self):
		pass


