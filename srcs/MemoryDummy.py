from collections import deque

class MemoryDummy():
	def __init__(self, config):
		self.config = config
		self.data = deque([], maxlen = config.capacity)



	def add(self, preprocessed_old_state, action, reward, 
			preprocessed_new_state, done, old_info, new_info):
		pass


	def get_memory(self):
		return self.data


	def clear_all(self):
		pass


	def get_batch(self):
		pass
		# might be uselesse as fuck


import random
import numpy as np

class DqnMemory():
	def __init__(self, config):
		self.config = config
		self.data = deque([], maxlen = config.capacity)



	def add(self, preprocessed_old_state, action, preprocessed_new_state, reward, done, old_info = None, new_info = None):
		self.data.append((preprocessed_old_state, action, preprocessed_new_state, reward, done, old_info, new_info))


	def get_memory(self):
		return self.data


	def sample(self, batch_size):
		return random.sample(self.data, batch_size)

	def clear_all(self):
		pass

	
	def __len__(self):
		return len(self.data)