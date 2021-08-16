from collections import deque
import numpy as np
import cv2
import logging

Logger = logging.getLogger("PreProc")
Logger.setLevel(logging.WARNING)
stream = logging.StreamHandler()
Logger.addHandler(stream)

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
		Logger.info(f"Processing.\nIn shape: {state.shape}")
		state = self.gray(state)
		state = self.resize(state, self.config.shrink_size)
		state = self.stack(state)
		Logger.debug(f"Out shape: {state.shape}")
		return state


	def resize(self, state, output_shape):
		return cv2.resize(state, dsize=output_shape, interpolation=cv2.INTER_CUBIC)


	def gray(self, state):
		return cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)

	
	def stack(self, state):
		self.skip_counter += 1
		self.skip_counter = self.skip_counter % self.config.frame_skip
		if (self.history[self.skip_counter] is None):
			self.history[self.skip_counter] = deque(maxlen = self.config.stack_size)
			for _ in range(self.config.stack_size):
				self.history[self.skip_counter].append(state.astype(np.float32))
		else:
			self.history[self.skip_counter].append(state.astype(np.float32))

		return np.stack(self.history[self.skip_counter], axis = 0)
		return np.array(self.history[self.skip_counter])


# from configDummy import config_Preprocessing
# p = Preprocessing(config_Preprocessing)
# st = (np.random.random((40, 40, 3)) * 255).astype(int)
# # stak = p.stack(st)
# pr = p.process(st)