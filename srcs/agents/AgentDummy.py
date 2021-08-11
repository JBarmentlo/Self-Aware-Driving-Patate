from MemoryDummy import DqnMemory
import numpy as np

class AgentDummy():
	def __init__(self, config):
		self.config = config
		self.memory = self._init_memory(config.config_Memory)
	

	def _init_memory(self, config = None):
		pass


	def get_action(self, state, episode = 0):
		return (np.random.random((2)) - [0.5, 0]) * [6, 1]

	
	def train(self):
		pass


import math
import random
import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
from collections import namedtuple, deque
# from itertools import count
# from PIL import Image
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


# env = gym.make('CartPole-v0').unwrapped

# # set up matplotlib
# is_ipython = 'inline' in matplotlib.get_backend()
# if is_ipython:
#     from IPython import display

# plt.ion()

# # if gpu is to be used





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALogger = logging.getLogger("DQNAgent")
ALogger.setLevel(logging.DEBUG)
stream = logging.StreamHandler()
ALogger.addHandler(stream)

class  DQNAgent():
	def __init__(self, config):
		self.config = config
		self.memory = self._init_memory(config.config_Memory)
		self.model = DQN(config)
		self.model.to(device)
		self.target_model = DQN(config)
		self.target_model.to(device)


	def steering_from_q_values(self, qs):
		ALogger.debug(f"Getting steering from qs: {qs}")
		bounds = self.config.action_space_boundaries[0]
		# np.array(qs.cpu())
		l = len(qs[0])
		idx = torch.argmax(qs[0])
		ALogger.debug(f"{idx = } {bounds[0] = }  {bounds[1] = } {l = }")
		out = (idx / l) * (bounds[1] - bounds[0]) + bounds[0]
		return out.item()


	def _init_memory(self, config = None):
		self.memory = DqnMemory(self.config.config_Memory)


	def get_action(self, state, episode = 0):
		return [self.steering_from_q_values(self.model.forward(torch.Tensor(state[np.newaxis, :, :]))), 0.3]

	
	def train(self):
		pass



def conv2d_size_out(size, kernel_size = 5, stride = 2):
			return (size - (kernel_size - 1) - 1) / stride  + 1


Logger = logging.getLogger("DQN")
Logger.setLevel(logging.WARN)
stream = logging.StreamHandler()
Logger.addHandler(stream)


class DQN(nn.Module):

	def __init__(self, config):
		super(DQN, self).__init__()
		in_channels = [*config.input_size][0]
		self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding = 2) # (kernal_size - 1) / 2 for same paddind
		self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding = 2)
		self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding = 1)
		# self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding = 2)
		self.flatten = nn.Flatten()
		self.dense1 = nn.Linear(6401, 512)
		self.dense2 = nn.Linear(512, *config.action_space_size) #TODO : GET from action space config



		# Number of Linear input connections depends on output of conv2d layers
		# and therefore the input image size, so compute it.
		# def conv2d_size_out(size, kernel_size = 5, stride = 2):
		# 	return (size - (kernel_size - 1) - 1) // stride  + 1
		# convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
		# convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
		# linear_input_size = convw * convh * 32
		# self.head = nn.Linear(linear_input_size, outputs)

	# Called with either one element to determine next action, or a batch
	# during optimization. Returns tensor([[left0exp,right0exp]...]).
	def forward(self, x):
		Logger.debug(f"Forward x: {x.shape}")
		x = x.to(device)
		x = F.relu(self.conv1(x))
		Logger.debug(f"conv1: {x.shape}")
		x = F.relu(self.conv2(x))
		Logger.debug(f"conv2: {x.shape}")
		x = F.relu(self.conv3(x))
		Logger.debug(f"conv3: {x.shape}")
		x = self.flatten(x)
		Logger.debug(f"Flat: {x.shape}")
		x = F.relu(self.dense1(x))
		Logger.debug(f"dense1: {x.shape}")
		x = self.dense2(x)
		Logger.debug(f"dense2: {x.shape}")
		return x