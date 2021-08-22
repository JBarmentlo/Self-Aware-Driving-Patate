from AutoEncoder import UndercompleteAutoEncoder
from ContractiveAutoEncoder import ContractiveAutoEncoder
from PoolingAutoEncoder import PoolingAutoEncoder
from PokAEmonTrainer import AutoEncoderTrainer

class DotDict(dict):
	"""
	a dictionary that supports dot notation 
	as well as dictionary access notation 
	usage: d = DotDict() or d = DotDict({'val1':'first'})
	set attributes: d.val2 = 'second' or d['val2'] = 'second'
	get attributes: d.val2 or d['val2']
	"""
	__getattr__ = dict.__getitem__
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

config_AutoEncoder = DotDict()
# Cache
config_AutoEncoder.model_dir			= "model_cache/autoencoder/"
config_AutoEncoder.train_dir			= "simulator_cache/"
config_AutoEncoder.test_dir				= "simulator_cache/"
config_AutoEncoder.name					= "Le_BG_du_13"
# Shapes
config_AutoEncoder.input_shape			= 12
config_AutoEncoder.output_shape			= 128
# Hyper Parameters
config_AutoEncoder.epochs				= 15
config_AutoEncoder.batch_size			= 64
config_AutoEncoder.lr					= 1e-3



if __name__ == "__main__":
	# type_ = "uae"
	type_ = "pae"
	# type_ = "cae"
	if type_ == "uae":
		ae = UndercompleteAutoEncoder(1, 10, learning_rate=1e-3)
	elif type_ == "cae":
		ae = ContractiveAutoEncoder(1, 10, learning_rate=1e-3)
	elif type_ == "pae":
		ae = PoolingAutoEncoder(config_AutoEncoder)

	AutoEncoderTrainer(ae, config_AutoEncoder, plot=True)