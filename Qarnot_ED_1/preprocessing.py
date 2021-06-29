import numpy as np
from config import config
import cv2
from PIL import Image, ImageOps
import sys
import os
import shutil
from config.env import keys
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import Augmentor

os.system('pip install -r requirements.txt')

from encoder import encod_model


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
		self.encoder = encod_model()
		self.encoder.summary()

	def rgb2gray(self, rgb):
		'''
		take a numpy rgb image return a new single channel image converted to greyscale
		'''
		return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
	
	def treatment_preproc(self, img_in):
		img = Image.fromarray(img_in, 'RGB')
		
		# print(img.size)
		width, height = img.size

		left = 1
		top = 40
		right = width
		bottom = 120
		
		# Cropped image of above dimension - It will crop the sky
		# (It will not change orginal image) 
		im1 = img.crop((left, top, right, bottom)) 

		basewidth = 80
		wpercent = (basewidth / float(img.size[0]))
		hsize = int((float(img.size[1]) * float(wpercent)))
		# im1 = im1.resize((basewidth, hsize), Image.ANTIALIAS)
		im1 = im1.resize((64, 64), Image.ANTIALIAS)
		# print(type(im1))
	
		im1_array = np.array(im1)
		im1_array = self.rgb2gray(im1_array)
		normalized_im1 = im1_array.astype('float32')/255.0
		# normalized_im1 = im1/255.0
		# Reshaping image, i.e. changing from (64, 64) to (64, 64, 1)
		normalized_im1 = np.expand_dims(normalized_im1,axis=-1)
		normalized_im1 = np.expand_dims(normalized_im1,axis=0)
		# img_ = np.expand_dims(normalized_im1, axis=0)
		# print("normalized_im1", normalized_im1.shape)
		
		encode_im1 = self.encoder.predict(normalized_im1)

		# print("encode_im1", encode_im1.shape)

		obs = np.array(im1) # ces 2 lignes sont utiles pour vérifier que le code fonctionne sans l'encoder
		obs = self.rgb2gray(obs) # return obs au lieu de encode_im1
		# print(obs.shape)
		# gray_image = ImageOps.grayscale(im1)
		return encode_im1
	
	def process_image(self, obs: np.ndarray) -> np.ndarray:
		"""Function to apply preprocessing to incomming observation to train the Patate

		Args:
			obs: Receive an observation of shape self.input_shape

		Returns:
			A preprocessed observation of shape self.output_shape

		"""
		# obs = self.rgb2gray(obs)
 
		obs = self.treatment_preproc(obs)
		# obs = cv2.resize(obs, self.output_shape)
		print("obs_cv2", obs.shape)

		return obs
