from collections import deque
import numpy as np
import cv2
import logging
import torch
from AutoEncoder import AutoEncoderTrainer, PoolingAutoEncoder

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
		# self.AutoEncoder = self._init_AutoEncoder()

	def _init_AutoEncoder(self):
		ae = PoolingAutoEncoder(self.config.config_AutoEncoder)
		if self.config.load_AutoEncoder:
			ae.load()
		else:
			AutoEncoderTrainer(ae, self.config.config_AutoEncoder)
		return ae

	def _autoencoder_prepare(self, state: np.array) -> torch.Tensor:
		"""
		Gets [H, W, C] to [1, C, H, W]
		"""
		state = np.moveaxis(state, -1, 0)
		state = np.expand_dims(state, 0)
		# Copy protects against the following Warning. 
		# Depending on performance losses we could ignore it
		state = np.copy(state)
		# 	UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  /pytorch/torch/csrc/utils/tensor_numpy.cpp:180.)
  		# 	state = torch.from_numpy(state).type(torch.float32) / 255.
		state = torch.from_numpy(state).type(torch.float32) / 255.
		return state

	def process(self, state):
		Logger.info(f"Processing.\nIn shape: {state.shape}")
		state = self.gray(state)
		state = self.resize(state, self.config.shrink_size)
		state = self.stack(state)
		# state = self._autoencoder_prepare(state)
		# state = self.AutoEncoder.encode(state)
		Logger.debug(f"Out shape: {state.shape}")
		return state

	def resize(self, state, output_shape):
		return cv2.resize(state, dsize=output_shape, interpolation=cv2.INTER_CUBIC)

	def gray(self, state):
		return cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)

	def colors_rounding(self, data: torch.Tensor, precision: int = 5) -> torch.Tensor:
		"""[summary]
		Reduce colors precision.

		Args:
			data (torch.Tensor):
				Simulator state, should be a matrix of floats between 0. and 1.
			precision (int, optional): 
				Define how many different values can take each channel, cant be higher than 255.
				Defaults to 5.

		Returns:
			(torch.Tensor): [description]
		"""
		max_value = 255.
		step = max_value / precision
		
		data *= max_value
		data = torch.div(data, step, rounding_mode="trunc")
		data *= step / max_value
		return data

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
