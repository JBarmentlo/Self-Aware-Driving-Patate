import logging
import pickle

import utils

Logger = logging.getLogger("Database")
Logger.setLevel(logging.INFO)
stream = logging.StreamHandler()
Logger.addHandler(stream)

class Database():
	def __init__(self, config, preprocessor = None):
		self.config				= config
		self.data				= []
		self.datapoints_counter	= 0
		self.upload_counter		= 0


	def add_point(self, point):
		if self.datapoints_counter + 1 > self.config.max_datapoints:
			self.upload()
		self.data.append(point)
		self.datapoints_counter += 1


	def upload(self):
		if self.config.s3 == True:
			pass
		else:
			file_name = f"{self.config.local_path}{self.upload_counter}"
			with open(file_name, "wb") as f:
				Logger.info(f"Saving data in file {file_name}")
				pickle.dump(self.data, f)
			self.upload_counter = 0
		self.datapoints_counter	= 0


	def download(self):
		pass


	def make_processed_memory(self):
		pass
		# self.download()
		# self.preprocessor.process(self.data)