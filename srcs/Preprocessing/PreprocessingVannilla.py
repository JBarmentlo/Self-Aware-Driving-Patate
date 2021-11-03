from collections import deque
import numpy as np
import cv2
import logging
import torch
from torchvision import datasets, transforms
from AutoEncoder import AutoEncoderTrainer, NiceAutoEncoder
from S3 import S3
from SimCache import SimCache
from ModelCache import ModelCache

Logger = logging.getLogger("PreProc")
Logger.setLevel(logging.INFO)
stream = logging.StreamHandler()
Logger.addHandler(stream)


class PreprocessingVannilla():
	def __init__(self, config, S3 = None):
		self.config  		= config
		self.history 		= [None for _ in range(config.frame_skip)]
		self.skip_counter 	= -1
		self.S3 = S3


	def process(self, state: np.ndarray) -> torch.Tensor:
		Logger.debug(f"Processing.\nIn shape: {state.shape}")
		state = self.gray(state)
		state = self.resize(state, self.config.shrink_size)
		state = torch.from_numpy(state)
		state = self.stack(state)

		state = state.cpu().detach()#.numpy()
		
		Logger.debug(f"Out shape: {state.shape}")
		return state


	def resize(self, state, output_shape):
		return cv2.resize(state, dsize=output_shape, interpolation=cv2.INTER_CUBIC)


	def gray(self, state):
		return cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)


	def stack(self, state):
		self.skip_counter += 1
		self.skip_counter = self.skip_counter % self.config.frame_skip
		state = np.array(state)
		if (self.history[self.skip_counter] is None):
			self.history[self.skip_counter] = deque(maxlen = self.config.stack_size)
			for _ in range(self.config.stack_size):
				self.history[self.skip_counter].append(state.astype(np.float32))
		else:
			self.history[self.skip_counter].append(state.astype(np.float32))

		return torch.Tensor(np.stack(self.history[self.skip_counter], axis = 0))
