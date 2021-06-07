import numpy as np
from config import config
import cv2
from PIL import Image, ImageOps
import sys
import os
import shutil
from encoder import encod_model

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Conv2DTranspose, Flatten
# from tensorflow.keras.layers import Reshape, BatchNormalization
# from tensorflow.keras.models import Model
# from tensorflow.keras import backend as K
# from tensorflow.keras.callbacks import TensorBoard

# def encod_model():


# 	image_width  = 64
# 	image_height = 64
# 	input_img = Input(shape=(image_width, image_height, 1))  

# 	# You can experiment with the encoder layers, i.e. add or change them
# 	x = Conv2D(32, (3, 3), activation='relu', strides=2, padding='same')(input_img)
# 	x = Conv2D(64, (3, 3), activation='relu', strides=2, padding='same')(x)

# 	# We need this shape later in the decoder, so we save it into a variable.
# 	# encoded_shape = K.int_shape(x)

# 	x = Flatten()(x)
# 	encoded = Dense(128)(x)
# 	# print("encoded_shape",encoded.shape)

# 	# Builing the encoder
# 	encoder = Model(input_img,encoded,name='encoder')
# 	# at this point the representation is 128-dimensional
# 	# encoder.summary()

# 	encoder.load_weights("./Qarnot_ED/output_2/encoder_weigths") 
	
# 	return encoder

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
		
		# Cropped image of above dimension 
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
