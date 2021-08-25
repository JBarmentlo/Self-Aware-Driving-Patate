import logging
import pickle

import utils

Logger = logging.getLogger("Database")
Logger.setLevel(logging.INFO)
stream = logging.StreamHandler()
Logger.addHandler(stream)

class Database():
	def __init__(self, config, s3 = None):
		self.config				= config
		self.data				= []
		self.datapoints_counter	= 0
		self.upload_counter		= 0
		self.s3 = s3


	def add_point(self, point):
		if self.config.on == False:
			return
		if self.datapoints_counter + 1 > self.config.max_datapoints:
			self.upload()
		self.data.append(point)
		self.datapoints_counter += 1
		return


	def upload(self, S3 = None):
		if self.config.on == False:
			return
		if self.s3:
			self.s3.upload_object(self.data, self.s3.config.s3_sim_path)
		else:
			file_name = f"{self.config.local_model_path}{self.upload_counter}"
			with open(file_name, "wb") as f:
				Logger.info(f"Saving data in file {file_name}")
				pickle.dump(self.data, f)
			self.upload_counter += 1
		self.datapoints_counter	= 0


	def download(self):
		pass


	def make_processed_memory(self):
		pass
		# self.download()
		# self.preprocessor.process(self.data)