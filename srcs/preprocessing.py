import os
# Qarnot Installation
# os.system('pip install -r requirements.txt')
import numpy as np
from numpy import asarray
from numpy.core.numerictypes import obj2sctype
from config import config
from PIL import Image, ImageOps
import sys
import shutil
# Token for Qarnot
# from config.env import keys
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import Augmentor

# from encodDecod import encod_model
from encodDecod import AutoEncoder

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

	def Image_crop(self, Im):
		""" 		
		Original Image size (120 x 160 x 3)
		Crop sky == 40 first lines 
	 	"""
		width, height = Im.size
		left = 1
		top = 40
		right = width
		bottom = 120
		# Cropped image of above dimension 
		# (It will not change orginal image) 
		img_crop = Im.crop((left, top, right, bottom))
		return img_crop
	
	def Image_resize(self, Im):
		""" Resize to obtain a square matrix """
		# config.img_rows, config.img_cols = 64, 64 for remind
		img_resize = Im.resize((config.img_rows, config.img_cols), Image.ANTIALIAS)
		return img_resize
	
	def rgb2gray(self, Im):
		'''
		take a numpy rgb image return a new single channel image converted to greyscale
		'''
		return ImageOps.grayscale(Im)
		
	def EncodedImage(self, gray_image, encoder):
		# Image Normalization
		normalized_im = np.array(gray_image).astype('float32')/255.0
		normalized_im = np.expand_dims(normalized_im,axis=0)
		# Encode image with the parameter encoder
		encode_im = encoder.predict(normalized_im)
		return encode_im

	def process_image(self, obs: np.ndarray, encoder) -> np.ndarray:
		"""Function to apply preprocessing to incomming observation to train the Patate

		Args:
			obs: Receive an observation of shape self.input_shape 

		Returns:
			A preprocessed observation of shape self.output_shape

		"""
		obs = self.Image_crop(obs)
		obs = self.Image_resize(obs)
		obs = self.rgb2gray(obs)
		obs = self.EncodedImage(obs, encoder)
		# print("obs_cv2", obs.shape)
		return obs

	def create_augmentor_pipeline(self, dir_path):
		""" Creates a pipeline for generating extra images from images at folder `dir_path`."""
		p = Augmentor.Pipeline(dir_path)
		p.resize(probability=1,width=64,height=64)
		p.rotate90(probability=0.1)
		p.rotate(probability=0.2, max_left_rotation=5, max_right_rotation=10)
		p.skew_left_right(probability=0.1)
		p.greyscale(probability=1)
		return p

	def preproc_AutoEncoder(self, Input_path, Output_Path):
		""" Images Pretreatment + Augmentor
		Images downloaded on Qarnot in the same directory as preprocessing   """
		# Format d'image en entrée : png
		# myPath = "./" "Sous qarnot le fichier d'exécution et le fichier input"
		for root, dirs, img in os.walk(Input_path):
			for name in img:
				if '.png' in name:
					img = Image.open(Input_path + '/' + name)
					obs = self.Image_crop(img)
					obs = self.Image_resize(obs)
					obs = self.rgb2gray(obs)
					name_wo_ext, ext = name.split('.') # Format image de sortie : jpg
					obs.save(Output_Path+ '/' + 'R_' + name_wo_ext +'.' + ext)
		p = self.create_augmentor_pipeline(Output_Path)

		# Generate 1000 images of (64 x 64) according to the pipeline
		#  and put them in `Output_path/output` folder
		num_samples = 1000
		p.sample(num_samples)


if __name__ == "__main__":
	if sys.argv[1] == "Preprocessing_AutoEncoder":
		Preproc = Preprocessing()
		Preproc.preproc_AutoEncoder("./task", "./output")