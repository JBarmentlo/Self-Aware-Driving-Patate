import io
import torch

import logging

ALogger = logging.getLogger(__file__)
ALogger.setLevel(logging.INFO)
stream = logging.StreamHandler()
ALogger.addHandler(stream)

class ModelCache():
	def __init__(self, conf_data, conf_s3, S3):
		self.conf_data = conf_data
		self.conf_s3 = conf_s3
		self.S3 = S3

	def _save_local(self, model, path):
		torch.save(self.model.state_dict(), path)
		
	def _save_s3(self, model, path):
		buffer = io.BytesIO()
		torch.save(model.state_dict(), buffer)
		# ! Reset read pointer. 
		# ! DOT NOT FORGET THIS, else all uploaded files will be empty!
		buffer.seek(0)
		self.S3.upload_bytes(buffer, f"{path}")

	def save(self, model, name):
		if self.conf_data.S3_connection == True:
			path = f"{self.conf_s3.model_folder}{name}"
			self._save_s3(model, path)
		else:
			path = f"{self.conf_data.local_model_folder}{name}"
			self._save_local(model, path)
	
	def _load_torch(self, model, where):
		model.load_state_dict(torch.load(where, map_location=torch.device('cpu'))) 
		model.eval()

	def _load_local(self, model, name):
		where = self.conf_data.local_model_folder + name
		self._load_torch(model, where)

	def _load_s3(self, model, name):
		bytes_obj = self.S3.get_bytes(self.conf_s3.model_folder + name)
		where = io.BytesIO(bytes_obj)
		self._load_torch(model, where)

	def load(self, model, name):
		try:
			if self.conf_data.S3_connection == True:
				self._load_s3(model, name)
			else:
				self._load_local(model, name)
			ALogger.info(f"Loaded model from file: {file_name}")
		except Exception as e:
			ALogger.error(f"You tried loading a model from file: {file_name} and this error occured: {e}")
