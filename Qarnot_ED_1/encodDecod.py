import os
os.system('pip install -r requirements.txt')
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import pickle
from datetime import datetime


def load_data(dir_path):
    """ Loads all the images from directory `dir_path`, converts them to matrices and return a list."""
    files = os.listdir(dir_path)
    num_files = len(files)
    data = []
    for f in tqdm(files, total=num_files):
        if '.jpg' in f:
            img = Image.open(os.path.join(dir_path,f))
            img_array = np.array(img)
            data.append(img_array)
    return data

"""> Note: The dataset contains several Farsi (Persian) characters written in `Moallah` font. It can replaced with any dataset of your interest."""

data = load_data(".")
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

# Compile and train the model. Log and visualize using tensorboard
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

h = autoencoder.fit(normalized_train_data, normalized_train_data,
                epochs=n_epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(normalized_test_data, normalized_test_data),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
encoder.save_weights("encoder_weigths") 
decoder.save_weights("decoder_weigths") 
autoencoder.save_weights("autoencoder_weigths")

# Plot the training history using altair

# import altair as alt
import pandas as pd

train_source = pd.DataFrame({'x':np.arange(0,n_epochs), 'y':h.history['loss'], 'orig_label': n_epochs * ['train_loss']})
val_source = pd.DataFrame({'x':np.arange(0,n_epochs), 'y':h.history['val_loss'], 'val_label': n_epochs * ['val_loss']})
legends = ['train loss', 'val loss']

# train_chart = alt.Chart(train_source).mark_line().encode(
#     alt.X('x', title='Epochs'),
#     alt.Y('y', title='Loss/Accuracy'),
#     color=alt.Color('orig_label:O', legend=alt.Legend(title=None))
# )
# val_chart = alt.Chart(val_source).mark_line().encode(
#     alt.X('x', title='Epochs'),
#     alt.Y('y', title='Loss/Accuracy'),
#     color=alt.Color('val_label:O', scale=alt.Scale(range=['red']), legend=alt.Legend(title=None))

    
# )
# alt.layer(train_chart, val_chart).resolve_scale(color='independent')

# plot the train and validation losses
N = np.arange(0, n_epochs)
# plt.figure()
# plt.plot(N, h.history['loss'], label='train_loss')
# plt.plot(N, h.history['val_loss'], label='val_loss')
# plt.title('Training Loss and Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Loss/Accuracy')
# plt.legend(loc='upper right')
# plt.show()
data_hist = [train_source, val_source, N,h.history['loss'], h.history['val_loss']]
with open("file_learning_curve.pk", "wb") as f:
    pickle.dump(data_hist, f)
    