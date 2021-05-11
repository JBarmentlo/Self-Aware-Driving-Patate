# Check autoencoding by compare sample image before and after auto-encoding 
import os
import numpy as np
import sklearn
import skimage
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import pickle
from skimage.metrics import structural_similarity as ssim

# Read autoencoder weights


def load_data(dir_path):
    """ Loads all the images from directory `dir_path`, converts them to matrices and return a list."""
    files = os.listdir(dir_path)
    num_files = len(files)
    data = []
    for f in tqdm(files, total=num_files):
        img = Image.open(os.path.join(dir_path,f))
        img_array = np.array(img)
        data.append(img_array)
    return data

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

data = load_data("./output1/output/")
# Split the dataset into 80% train and 20% test sets.
# keep random_state identical in VisuEncoding and Encoding 
train_data,test_data,_,_ = train_test_split(data,data,test_size=0.2, random_state = 1042 )
train_data = np.array(train_data)
test_data = np.array(test_data)


# Normalizing train and test data
normalized_train_data = train_data.astype('float32')/255.0
normalized_test_data = test_data.astype('float32')/255.0

# Reshaping train and test sets, i.e. changing from (64, 64) to (64, 64, 1)
normalized_train_data = np.expand_dims(normalized_train_data,axis=-1)
normalized_test_data = np.expand_dims(normalized_test_data,axis=-1)
print('Normalization and reshaping is done.')
print('Input shape = {}'.format(normalized_train_data.shape[1:]))
"""### Defining the Encoder

"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Conv2DTranspose, Flatten
from tensorflow.keras.layers import Reshape, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard

image_width  = 64
image_height = 64
n_epochs     = 15
batch_size   = 128

input_img = Input(shape=(image_width, image_height, 1))  

# You can experiment with the encoder layers, i.e. add or change them
x = Conv2D(32, (3, 3), activation='relu', strides=2, padding='same')(input_img)
x = Conv2D(64, (3, 3), activation='relu', strides=2, padding='same')(x)

# We need this shape later in the decoder, so we save it into a variable.
encoded_shape = K.int_shape(x)

x = Flatten()(x)
encoded = Dense(128)(x)

# Builing the encoder
encoder = Model(input_img,encoded,name='encoder')

# at this point the representation is 128-dimensional
encoder.summary()

"""### Defining the Decoder"""

# Input shape for decoder
encoded_input = Input(shape=(128,))
x = Dense(np.prod(encoded_shape[1:]))(encoded_input)
x = Reshape((encoded_shape[1], encoded_shape[2], encoded_shape[3]))(x)
x = Conv2DTranspose(64,(3, 3), activation='relu',strides=2, padding='same')(x)
x = Conv2DTranspose(32,(3, 3), activation='relu', strides=2, padding='same')(x)
x = Conv2DTranspose(1,(3, 3), activation='sigmoid', padding='same')(x)

decoder = Model(encoded_input,x,name='decoder')
decoder.summary()

"""### Defining the Autoencoder"""

autoencoder = Model(input_img, decoder(encoder(input_img)),name="autoencoder")
autoencoder.summary()

encoder.load_weights("./output_2/encoder_weigths") 
decoder.load_weights("./output_2/decoder_weigths") 
autoencoder.load_weights("./output_2/autoencoder_weigths")

#hide
# Plot the training history using altair
# load learning history

data_hist = pickle.load( open( "./output_2/file_learning_curve.pk", "rb" ) )
# data_hist = [train_source, val_source, N,h.history['loss'], h.history['val_loss']]
train_source = data_hist[0]
val_source = data_hist[1]
N = data_hist[2]
h_history_loss = data_hist[3]
h_history_val_loss = data_hist[4]

import altair as alt
import pandas as pd

legends = ['train loss', 'val loss']

train_chart = alt.Chart(train_source).mark_line().encode(
    alt.X('x', title='Epochs'),
    alt.Y('y', title='Loss/Accuracy'),
    color=alt.Color('orig_label:O', legend=alt.Legend(title=None))
)
val_chart = alt.Chart(val_source).mark_line().encode(
    alt.X('x', title='Epochs'),
    alt.Y('y', title='Loss/Accuracy'),
    color=alt.Color('val_label:O', scale=alt.Scale(range=['red']), legend=alt.Legend(title=None))
)

# plot the train and validation losses
# N = np.arange(0, n_epochs)
plt.figure()
plt.plot(N, h_history_loss, label='train_loss')
plt.plot(N, h_history_val_loss, label='val_loss')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='upper right')
plt.show()

# Make predictions on the test set
decoded_imgs = autoencoder.predict(normalized_test_data)

def visualize(model, X_test, n_samples): 
    """ Visualizes the original images and the reconstructed ones for `n_samples` examples 
    on the test set `X_test`."""
      
    # Reconstructing the encoded images 
    reconstructed_images = model.predict(X_test)

    plt.figure(figsize =(20, 4)) 
    for i in range(1, n_samples): 
          
        # Generating a random to get random results 
        rand_num = np.random.randint(0, 200) 
      
        # To display the original image 
        mse_img = mse(X_test[rand_num].reshape(image_width, image_width), reconstructed_images[rand_num].reshape(image_width, image_width))
        ssim_img = ssim(X_test[rand_num].reshape(image_width, image_width), reconstructed_images[rand_num].reshape(image_width, image_width))        # plt.suptitle("MSE: %.2f, SSIM: %.2f" % (mse_img, mse_img))
        ax = plt.subplot(2, 10, i)
        title = ax.title.set_text("MSE: %.2f, SSIM: %.2f" % (mse_img, ssim_img))
        ax.title.set_fontsize('6')
        plt.imshow(X_test[rand_num].reshape(image_width, image_width)) 
        plt.gray() 
        ax.get_xaxis().set_visible(False) 
        ax.get_yaxis().set_visible(False) 
  
        # To display the reconstructed image 
        ax = plt.subplot(2, 10, i + 10) 
        plt.imshow(reconstructed_images[rand_num].reshape(image_width, image_width)) 
        plt.gray() 
        ax.get_xaxis().set_visible(False) 
        ax.get_yaxis().set_visible(False) 


    # Displaying the plot 
    plt.show()


# Plots `n_samples` images. Top row is the original images and the lower row is the reconstructed ones.
n_samples = 10
visualize(autoencoder,normalized_test_data, n_samples)

