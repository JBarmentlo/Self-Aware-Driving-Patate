from collections import deque
import numpy as np

class PreprocessingDummy():
	def __init__(self, config):
		self.config  = config

	
	def process(self, state):
		return state


class Preprocessing():
	def __init__(self, config):
		self.config  		= config
		self.history 		= [None for _ in range(config.frame_skip)]
		self.skip_counter 	= -1

	
	def process(self, state):
		return state

	
	def stack_state(self, state):
		self.skip_counter += 1
		self.skip_counter = self.skip_counter % self.config.frame_skip
		if (self.history[self.skip_counter] is None):
			self.history[self.skip_counter] = deque(maxlen = self.config.stack_size)
			for _ in range(self.config.stack_size):
				self.history[self.skip_counter].append(state)
		else:
			self.history[self.skip_counter].append(state)

		return np.stack(self.history[self.skip_counter], axis = self.history[self.skip_counter][0].ndim)
		return np.array(self.history[self.skip_counter])


from configDummy import config_Preprocessing
p = Preprocessing(config_Preprocessing)
st = np.random.random((1,2, 3))
stak = p.stack_state(st)
d = p.history[0]