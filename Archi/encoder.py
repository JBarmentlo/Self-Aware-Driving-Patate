import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Conv2DTranspose, Flatten
from tensorflow.keras.layers import Reshape, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from utils_get_abs_path import get_path_to_cache

def encod_model():

	image_width  = 64
	image_height = 64
	input_img = Input(shape=(image_width, image_height, 1))  

	# You can experiment with the encoder layers, i.e. add or change them
	x = Conv2D(32, (3, 3), activation='relu', strides=2, padding='same')(input_img)
	x = Conv2D(64, (3, 3), activation='relu', strides=2, padding='same')(x)

	# We need this shape later in the decoder, so we save it into a variable.
	# encoded_shape = K.int_shape(x)

	x = Flatten()(x)
	encoded = Dense(128)(x)
	# print("encoded_shape",encoded.shape)

	# Builing the encoder
	encoder = Model(input_img,encoded,name='encoder')
	# at this point the representation is 128-dimensional
	# encoder.summary()

	cache_path = get_path_to_cache("model_cache/encoder/encoder_weigths")
	encoder.load_weights(cache_path) 
	
	return encoder