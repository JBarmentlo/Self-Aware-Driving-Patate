import random
import numpy as np
from torch.utils.data import Dataset
from collections import deque

class Memory():
	def __init__(self, config):
		self.config = config
		self.data = deque([], maxlen = config.capacity)



	def add(self, data):
		self.data.append(data)


	def get_memory(self):
		return self.data


	def clear_all(self):
		pass


	def get_batch(self):
		pass
		# might be uselesse as fuck

	def __len__(self):
		return len(self.data)

	def __getitem__(self, i):
		return self.data[i]

class AutoEncoderDataset(Dataset):
	def __init__(self):
		self.data = []

	def add(self, data):
		self.data.append(data)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, i):
		return self.data[i]


class SACDataset(Dataset):
	def __init__(self):
		self.data = []

	def add(self, processed_state, action, new_processed_state, reward, done):
		data = [processed_state, action, new_processed_state, reward, done]
		self.data.append(data)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, i):
		return self.data[i]

class DqnMemory(Dataset):
	def __init__(self, config):
		self.config = config
		self.data = deque([], maxlen = config.capacity)



	def add(self, preprocessed_old_state, action, preprocessed_new_state, reward, done):
		self.data.append((preprocessed_old_state, action, preprocessed_new_state, reward, done))


	def get_memory(self):
		return self.data


	def sample(self, batch_size):
		return random.sample(self.data, batch_size)

	def clear_all(self):
		pass

	def __len__(self):
		return len(self.data)

	def __getitem__(self, i):
		return self.data[i]