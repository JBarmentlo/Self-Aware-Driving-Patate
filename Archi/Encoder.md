Version with images loaded on your computer

Create directories :
/Self-Aware-Driving-Patate/Archi/Task
/Self-Aware-Driving-Patate/Archi/output
/Self-Aware-Driving-Patate/model_cache/autoencoder
/Self-Aware-Driving-Patate/model_cache/decoder
/Self-Aware-Driving-Patate/model_cache/encoder
/Self-Aware-Driving-Patate/pickle_archive

File preprocessing.py
Images stored in 'task' directory

First treatment :
- crop the sky
- resize to 64x64
- grey colorising
- increase number of images by rotation
Images stored in 'output/output' ===> 1000 images generated : paramètres à placer dans config.py
command : python preprocessing.py "Preprocessing_AutoEncoder"

Autoencoder training
File encodDecoded.py
command python encodDecod.py 'Training_AutoEncoder'

Train the autoencoder and return encoder, decoder and autoencoder and weights stored in ../cache_model/encoder/, /decoder/ and /autoencoder/ directories

command python encodDecod.py 'Training_AutoEncoder' "Convergence Visualization"
Allow the Visualization of the learning curve and comparison of 10 images form test dataset with autoencoded images
with metrics 'similarity' and Root Mean Square Error 


For the train simulator 
* Encoded Image
size of the vector set in file config.py : config.output_shape =128 by default
Train simulator activate encoded image in Class NeuralPlayer
Load : 
		self.AC = AutoEncoder()
		self.encoder, _, _ = self.AC.AutoEncoder_model(config.img_rows, config.img_cols)
		self.enc_loaded = self.AC.Loaded_Encoder("", self.encoder)
Method "def Loaded_Encoder(self, weight_path, encoder)"
weight_path by default : "../model_cache/encoder"

In method prepare_state :
Input : image as numpyarray, state in 
		state = Image.fromarray(state) # to check
		x_t = np.array(self.preprocessing.process_image(state, self.enc_loaded))



