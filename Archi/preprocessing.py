import numpy as np
from config import config
import cv2

class Preprocessing():
	def __init__(self, input_shape: tuple = None, output_shape: tuple = None):
		if not input_shape:
			self.input_shape = (config.sim_img_rows,
							   config.sim_img_cols,
							   config.sim_img_channels)
		else:
			self.input_shape = input_shape
		if not output_shape:
			self.output_shape = (config.prep_img_rows,
								config.prep_img_cols)
		else:
			self.output_shape = output_shape

	def rgb2gray(self, rgb):
		'''
		take a numpy rgb image return a new single channel image converted to greyscale
		'''
		return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

	def process_image(self, obs: np.ndarray) -> np.ndarray:
		"""Function to apply preprocessing to incomming observation to train the Patate

		Args:
			obs: Receive an observation of shape self.input_shape

		Returns:
			A preprocessed observation of shape self.output_shape

		"""
		obs = self.rgb2gray(obs)
		obs = cv2.resize(obs, self.output_shape)
		return obs
