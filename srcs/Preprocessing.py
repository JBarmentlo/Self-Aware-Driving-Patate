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
Logger.setLevel(logging.WARNING)
stream = logging.StreamHandler()
Logger.addHandler(stream)


class Preprocessing():
	def __init__(self, config):
		self.config  		= config
		self.history 		= [None for _ in range(config.frame_skip)]
		self.skip_counter 	= -1

		self.AutoEncoder = self._init_AutoEncoder()

	def _init_AutoEncoder(self):
		my_S3 = None
		conf_data = self.config.config_AutoEncoder.config_AE_Datasets
		if conf_data.S3_connection == True:
			my_S3 = S3(conf_data.config_S3)
		mc = ModelCache(conf_data, conf_data.config_S3, my_S3)
		ae = NiceAutoEncoder(self.config.config_AutoEncoder, mc)
		if self.config.load_AutoEncoder:
			ae.load()
		else:
			my_SimCache = SimCache(conf_data, my_S3)
			AutoEncoderTrainer(ae,
					self.config.config_AutoEncoder,
					plot=True,
					SimCache=my_SimCache,
					Prepocessing=self)
		return ae

	def _autoencoder_prepare(self, state: np.array, extand=True) -> torch.Tensor:
		"""
		Gets [H, W, C] to [1, C, H, W]
		"""
		state = np.moveaxis(state, -1, 0)
		if extand:
			state = np.expand_dims(state, 0)
		# Copy protects against the following Warning. 
		# Depending on performance losses we could ignore it
		state = np.copy(state)
		# 	UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  /pytorch/torch/csrc/utils/tensor_numpy.cpp:180.)
  		# 	state = torch.from_numpy(state).type(torch.float32) / 255.
		state = torch.from_numpy(state).type(torch.float32) / 255.
		return state

	def before_AutoEncoder(self, state: np.ndarray, training:bool=False) -> torch.Tensor:
		# print(f"{state = }")
		# print(f"{state.shape = }")
		state = self._autoencoder_prepare(state, extand=not training)
		transform = transforms.Compose([
				transforms.Resize(self.config.shrink_size, transforms.InterpolationMode.BICUBIC)])
		state = transform(state)
		# print(f"{state.shape = }")
		return state
	
	def after_AutoEncoder(self, state: torch.Tensor) -> torch.Tensor:
		state = self.stack(state)
		return state

	def process(self, state: np.ndarray) -> torch.Tensor:
		Logger.info(f"Processing.\nIn shape: {state.shape}")
		# print(f"Prep IN {state.shape = }")
		# state = self.gray(state)
		# state = self.resize(state, self.config.shrink_size)
		# state = self.stack(state)
		state = self.before_AutoEncoder(state, training=False)
		state = self.AutoEncoder.encode(state)
		state = self.after_AutoEncoder(state)

		state = state.cpu().detach()#.numpy()
		
		# print(f"Prep OUT {state.shape = }") 
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

	def stack(self, state: torch.Tensor) -> torch.Tensor:
		self.skip_counter += 1
		self.skip_counter = self.skip_counter % self.config.frame_skip
		if (self.history[self.skip_counter] is None):
			self.history[self.skip_counter] = deque(maxlen = self.config.stack_size)
			for _ in range(self.config.stack_size):
				self.history[self.skip_counter].append(state.to(torch.float32))
		else:
			self.history[self.skip_counter].append(state.to(torch.float32))
		
		stacked_state = torch.stack(tuple(self.history[self.skip_counter]), axis = 0)
		stacked_state = torch.moveaxis(stacked_state, 1, 0)
		return stacked_state
