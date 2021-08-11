class MemoryDummy():
	def __init__(self, config):
		self.config = config
		self.data = None



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