import logging
import pickle

import utils

Logger = logging.getLogger("SimCache")
Logger.setLevel(logging.INFO)
stream = logging.StreamHandler()
Logger.addHandler(stream)

class SimCache():
	def __init__(self, config, S3 = None):
		self.config				= config
		self.data				= []
		self.datapoints_counter	= 0
		self.upload_counter		= 0
		self.S3 = S3


	def _reset(self):
		self.datapoints_counter	= 0
		self.data = []


	def add_point(self, point):
		self.data.append(point)
		self.datapoints_counter += 1


	def upload(self, S3 = None):
		if self.config.S3_connection == True:
			self.S3.upload_bytes(self.data, self.S3.config.S3_sim_path)
		else:
			file_name = f"{self.config.sim_infos_name}{self.upload_counter}"
			with open(file_name, "wb") as f:
				Logger.info(f"Saving Simulator Cache in file: {file_name}")
				pickle.dump(self.data, f)
			self.upload_counter += 1
		self._reset()


	def download(self):
		pass


	def make_processed_memory(self):
		pass