import os
# os.system('pip install -r requirements.txt') # Qarnot install
import sys
import numpy as np
import pandas as pd
import altair as alt
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import pickle
from skimage.metrics import structural_similarity as ssim
from config import config
from utils_get_abs_path import get_path_to_cache
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Conv2DTranspose, Flatten
from tensorflow.keras.layers import Reshape, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard

class AutoEncoder():
    def __init__(self, input_shape: tuple = None, autoencoder_active = True):
        if not input_shape:
            self.input_shape = (config.sim_img_rows, config.sim_img_cols, config.sim_img_channels)
        else:
            self.input_shape = input_shape
        if not autoencoder_active:
            self.output_shape = (config.prep_img_rows,
								config.prep_img_cols)
        else:
            self.output_shape = config.output_shape
        print("output_shape",self.output_shape)

    def load_data(self, dir_path):
        """ Loads all the images from directory `dir_path`, converts them to matrices and return a list."""
        files = os.listdir(dir_path)
        num_files = len(files)
        data = []
        for f in tqdm(files, total=num_files):
            if '.png' in f:
                img = Image.open(os.path.join(dir_path,f))
                img_array = np.array(img)
                data.append(img_array)
        return data

        ### Defining the Encoder
    def AutoEncoder_model(self, image_width,image_height):
        """create a autoencoder model with fixed architecture
        decomposed in autoencoder = decoder(encoder) 
        input size of images to autoencode
        return 3 models  """
        
        input_img = Input(shape=(image_width, image_height, 1)) 
        output_shape_encoded = self.output_shape
        # You can experiment with the encoder layers, i.e. add or change them
        x = Conv2D(32, (3, 3), activation='relu', strides=2, padding='same')(input_img)
        x = Conv2D(64, (3, 3), activation='relu', strides=2, padding='same')(x)

        # We need this shape later in the decoder, so we save it into a variable.
        encoded_shape = K.int_shape(x)

        x = Flatten()(x)
        encoded = Dense(output_shape_encoded)(x)
        # print("encoded_shape",encoded.shape)

        # Builing the encoder
        self.encoder = Model(input_img,encoded,name='encoder')
        # at this point the representation is 'output_shape_encoded'-dimensional
        # encoder.summary()
         
        """### Defining the Decoder"""
        # Input shape for decoder
        encoded_input = Input(shape=(output_shape_encoded,))
        x = Dense(np.prod(encoded_shape[1:]))(encoded_input)
        x = Reshape((encoded_shape[1], encoded_shape[2], encoded_shape[3]))(x)
        x = Conv2DTranspose(64,(3, 3), activation='relu',strides=2, padding='same')(x)
        x = Conv2DTranspose(32,(3, 3), activation='relu', strides=2, padding='same')(x)
        x = Conv2DTranspose(1,(3, 3), activation='sigmoid', padding='same')(x)

        self.decoder = Model(encoded_input,x,name='decoder')
        self.decoder.summary()

        """### Defining the Autoencoder"""
        self.autoencoder = Model(input_img, self.decoder(self.encoder(input_img)),name="autoencoder")
        self.autoencoder.summary()

        return self.encoder, self.decoder, self.autoencoder

   
    def Loaded_Encoder(self, weight_path, encoder):
        """Given encoder model architecture after training have been done
        the function load weights stored in weight_path
         and return encoder that is able to predict """
        if weight_path =="":
            weight_path = get_path_to_cache("model_cache/encoder/encoder_weights")
        encoder.load_weights(weight_path)
        return encoder

    def Prepare_input_data(self, data):
        """ Load preprocessed data to be trained by autoencoder and used 
        to visualize efficiency of autoencoder
         """
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

        return normalized_train_data, normalized_test_data

        # METRICS - VISUALIZATION ERROR
    def mse(self, imageA, imageB):
        # the 'Mean Squared Error' between the two images is the
        # sum of the squared difference between the two images;
        # NOTE: the two images must have the same dimension
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])
        
        # return the MSE, the lower the error, the more "similar"
        # the two images are
        return err

    def visualize(self, file_autoencoder_learning_curve, n_samples, model, weight_path_autoencoder, X_test):

        "Load data of convergence"
        Lrn_file_pickle_path = get_path_to_cache("/model_cache/pickle_archive")
        data_hist = pickle.load( open(Lrn_file_pickle_path + "/" + file_autoencoder_learning_curve, "rb" ) )
        "load weitghts in autoencoder"
        if weight_path_autoencoder =="":
            weight_path_autoencoder = get_path_to_cache("/model_cache/autoencoder/autoencoder_weights")
            model.load_weights(weight_path_autoencoder)

        train_source = data_hist[0]
        val_source = data_hist[1]
        N = data_hist[2]
        h_history_loss = data_hist[3]
        h_history_val_loss = data_hist[4]

        # plot the train and validation losses
        plt.figure()
        plt.plot(N, h_history_loss, label='train_loss')
        plt.plot(N, h_history_val_loss, label='val_loss')
        plt.title('Training Loss and Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Loss/Accuracy')
        plt.legend(loc='upper right')
        plt.show()

        # # Make predictions on the test set

            # Reconstructing the encoded images 
        reconstructed_images = model.predict(X_test)

        plt.figure(figsize =(20, 4)) 
        for i in range(1, n_samples): 
          
            # Generating a random to get random results 
            rand_num = np.random.randint(0, 200) 
        
            # To display the original image 
            mse_img = self.mse(X_test[rand_num].reshape(image_width, image_width), reconstructed_images[rand_num].reshape(image_width, image_width))
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

if __name__ == "__main__":
    if sys.argv[1] == "Training_AutoEncoder":
        AC = AutoEncoder()
        data = AC.load_data("./output/output")
        print(len(data))
        # Split the dataset into 80% train and 20% test sets.
        # keep random_state identical in VisuEncoding and Encoding 

        normalized_train_data, normalized_test_data = AC.Prepare_input_data(data)

        # Compile and train the model. Log and visualize using tensorboard

        image_width  = config.prep_img_rows
        image_height = config.prep_img_cols
          

        encoder, decoder, aec = AC.AutoEncoder_model(image_width, image_height)
        
        aec.compile(optimizer='adam', loss='binary_crossentropy')
        epochs=config.epochs
        batch_size=config.batch_size
        
        h = aec.fit(normalized_train_data, normalized_train_data,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(normalized_test_data, normalized_test_data),
                        callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
        weight_path_encoder = get_path_to_cache("./model_cache/encoder/")
        weight_path_decoder = get_path_to_cache("./model_cache/decoder/")
        weight_path_autoencoder = get_path_to_cache("./model_cache/autoencoder/")
        encoder.save_weights(weight_path_encoder +'/encoder_weights') 
        decoder.save_weights(weight_path_decoder +'/decoder_weights') 
        aec.save_weights(weight_path_autoencoder +'/autoencoder_weights')

        # reconstructed_images = aec.predict(normalized_test_data)
        # weight_path = get_path_to_cache("./model_cache/autoencoder/autoencoder_weights") 
        # aec.load_weights(weight_path)


        # Store convergence and visualisation data
        train_source = pd.DataFrame({'x':np.arange(0,epochs), 'y':h.history['loss'], 'orig_label': epochs * ['train_loss']})
        val_source = pd.DataFrame({'x':np.arange(0,epochs), 'y':h.history['val_loss'], 'val_label': epochs * ['val_loss']})
        legends = ['train loss', 'val loss']
        N = np.arange(0, epochs)
        data_hist = [train_source, val_source, N,h.history['loss'], h.history['val_loss']]
        pickle_archive_path = get_path_to_cache("/model_cache/pickle_archive")
        with open(pickle_archive_path+"/file_autoencoder_learning_curve.pk", "wb") as f:
            pickle.dump(data_hist, f)

        
        if len(sys.argv) ==3 and sys.argv[2] == "Convergence Visualization":
            # charger les weights dans autoencoder
            # AC = AutoEncoder()
            n_samples = 10 # Sample of images compared
            # weight_path_autoencoder = get_path_to_cache("./model_cache/autoencoder/")
            AC.visualize("file_autoencoder_learning_curve.pk", n_samples, aec, "already_loaded", normalized_test_data)
        
        # Convergence Vizualisation later
    if sys.argv[1] == "Convergence Visualization":
        image_width  = config.prep_img_rows
        image_height = config.prep_img_cols
        AC = AutoEncoder()
        encoder, decoder, aec = AC.AutoEncoder_model(image_width,image_height)
        data = AC.load_data("./output/output")
        print(len(data))
        # Split the dataset into 80% train and 20% test sets.
        # keep random_state identical in VisuEncoding and Encoding 

        normalized_train_data, normalized_test_data = AC.Prepare_input_data(data)
        weight_path = get_path_to_cache("/model_cache/autoencoder/autoencoder_weights")

        aec.load_weights(weight_path)

        n_samples = 10 # Sample of images compared
        AC.visualize("file_autoencoder_learning_curve.pk", n_samples, aec, "already_loaded", normalized_test_data)