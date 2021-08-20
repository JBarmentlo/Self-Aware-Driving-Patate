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
import random
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
import utils

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
		if (self.config.load_model):
			self._load_model(self.config.model_path)
		self.model.to(device)
		self.target_model = DQN(config)
		self.target_model.to(device)
		self.target_model.load_state_dict(self.model.state_dict())
		self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
		self.criterion = nn.MSELoss()
		self.update_target_model_counter = 0


	def _load_model(self, path):
		ALogger.info(f"Loading model from {path}")
		try:
			if (path is not None):
				self.model.load_state_dict(torch.load(path))
				self.model.eval()
		except Exception as e:
			ALogger.error(f"You tried loading a model from path: {path} and this error occured: {e}")


	def action_from_q_values(self, qs):
		ALogger.debug(f"Getting steering from qs: {qs}")
		bounds = self.config.action_space_boundaries[0]
		# np.array(qs.cpu())
		l = len(qs[0])
		idx = torch.argmax(qs[0])
		action = self.config.action_space[idx]
		return action


	def _update_target_model(self):
		self.target_model.load_state_dict(self.model.state_dict())


	def _init_memory(self, config = None):
		return DqnMemory(self.config.config_Memory)


	def _update_epsilon(self):
			if self.config.epsilon > self.config.epsilon_min:
				self.config.epsilon -= (self.config.initial_epsilon - self.config.epsilon_min) / self.config.steps_to_eps_min
				ALogger.info(f"Updating agent epsilon to {self.config.epsilon}")



	def get_action(self, state, episode = 0):
		# TODO: UPDATE THIS FOR UNFIXED THROTTLE
		if np.random.rand() > self.config.epsilon :
			ALogger.debug(f"Not Random action being picked")
			action = [self.action_from_q_values(self.model.forward(torch.Tensor(state[np.newaxis, :, :]))), 1.0]
			ALogger.debug(f"{action = }")
			return action
		else:
			ALogger.debug(f"Random action being picked")
			action = random.choice(self.config.action_space)
			ALogger.debug(f"{action = }")
			return action


	def train_model(self, x, y):
		self.model.train()
		y_hat = self.model.forward(x)
		loss = self.criterion(y_hat, y)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		self.model.eval()

	
	def replay_memory(self):
		ALogger.info(f"Replay from memory")
		if len(self.memory) < self.config.min_memory_size:
			return
		
		self.update_target_model_counter += 1
		self.optimizer.zero_grad()

		batch_size = min(self.config.batch_size, len(self.memory))
		# batch = self.memory.sample(batch_size)
		train_dataloader = DataLoader(self.memory, batch_size=batch_size, shuffle=True)
		batch = next(iter(train_dataloader)) # s, a, s', r, d
		s = batch[0].to(torch.float32)
		ss = batch[2].to(torch.float32)
		a = batch[1]
		for i in range(len(a)):
			a[i] = a[i].to(torch.float32)
		r = batch[3].to(torch.float32)
		d = batch[4]
		d = ~d
		d = d.to(torch.int64)
		ALogger.debug(f"{s.size() = } {ss.size() = } {a[0].size() = } {a[1].size() = } {r.size() = }")
		qs = self.model.forward(s).cpu()
		qss = self.target_model.forward(ss).cpu()
		qss_max, _  = torch.max(qss, dim = 1)
		action_space_size = [*self.config.action_space_size]
		a_bin = utils.val_to_bin(a[0], self.config.action_space_boundaries[0], action_space_size[0]).to(torch.int64)
		a2_bin = utils.val_to_bin(a[1], self.config.action_space_boundaries[1], action_space_size[1]).to(torch.int64)
		bin = a_bin * (a2_bin + 1)
		ALogger.debug(f"{a[0] = } {a_bin = } \n{a[1] = } {a2_bin = } \n{bin =}")
		# y.scatter(bin)
		hot = torch.nn.functional.one_hot(bin, action_space_size[1] * action_space_size[0])
		ALogger.debug(f"{hot = }")
		targets = qs.clone()
		targets = targets.detach()
		targets = hot * (r.view(batch_size, -1) + self.config.lr * (qss_max.view(batch_size, -1) * d.view(batch_size, -1))) - hot * qs + qs
		ALogger.debug(f"{qs = } {targets = }")

		error = self.criterion(qs, targets).to(torch.float32)
		error.backward()
		self.optimizer.step()
		if (self.update_target_model_counter % self.config.target_model_update_frequency == 0):
			self._update_target_model


	def train(self):
		pass



def conv2d_size_out(size, kernel_size = 5, stride = 2):
			return (size - (kernel_size - 1) - 1) / stride  + 1


Logger = logging.getLogger("DQN")
Logger.setLevel(logging.INFO)
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
		self.dense1 = nn.Linear(1600, 512)
		output_size = 1
		for s in config.action_space_size:
			output_size *= s
		self.dense2 = nn.Linear(512, *config.action_space_size)


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