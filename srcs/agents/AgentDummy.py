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
from torch.utils.data import DataLoader


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
		self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
		self.criterion = nn.MSELoss()



	def steering_from_q_values(self, qs):
		ALogger.debug(f"Getting steering from qs: {qs}")
		bounds = self.config.action_space_boundaries[0]
		# np.array(qs.cpu())
		l = len(qs[0])
		idx = torch.argmax(qs[0])
		ALogger.debug(f"{idx = } {bounds[0] = }  {bounds[1] = } {l = }")
		out = (idx / l) * (bounds[1] - bounds[0]) + bounds[0]
		return out.item()


	def update_target_model(self):
		self.target_model.load_state_dict(self.model.state_dict())


	def _init_memory(self, config = None):
		return DqnMemory(self.config.config_Memory)


	def _update_epsilon(self):
			if self.epsilon > self.config.epsilon_min:
				self.epsilon -= (self.config.initial_epsilon - self.config.epsilon_min) / self.config.steps_to_eps_min


	def get_action(self, state, episode = 0):
		if np.random.rand() > self.config.epsilon :
			ALogger.debug(f"Not Random action beign picked")
			return [self.steering_from_q_values(self.model.forward(torch.Tensor(state[np.newaxis, :, :]))), 0.3]
		else:
			ALogger.debug(f"Random action beign picked")
			return [np.random.choice(self.config.action_space[i], 1)[0] for i in range(len(self.config.action_space_size))]


	def train_model(self, x, y):
		self.model.train()
		y_hat = self.model.forward(x)
		loss = self.criterion(y_hat, y)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		self.model.eval()

	
	def replay_memory(self):
		if len(self.memory) < self.config.min_memory_size:
			return
		
		batch_size = min(self.config.batch_size, len(self.memory))
		# batch = self.memory.sample(batch_size)
		train_dataloader = DataLoader(self.memory, batch_size=batch_size, shuffle=True)
		batch = next(iter(train_dataloader))
		return batch

		# state, action, new_state, reward, done, old_info, new_info = zip(batch) #* wierd ?
		# s = np.concatenate(state)
		# ss = np.concatenate(new_state)
		# targets = self.model.forward(s)


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
		self.conv0 = nn.Conv2d(in_channels, 24, kernel_size=5, stride=2, padding = 2) # (kernal_size - 1) / 2 for same paddind
		self.conv1 = nn.Conv2d(24, 32, kernel_size=5, stride=2, padding = 2) # (kernal_size - 1) / 2 for same paddind
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
		x = F.relu(self.conv0(x))
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