import numpy as np
from utils import get_path_to_cache
import uuid
from datetime import datetime

date = f"{datetime.now().day}_{datetime.now().month}.{datetime.now().hour}_{datetime.now().minute}"


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

# TODO: Ask for automated find if os.environ["SIM_PATH"] doasnt exist (needed by server)

config = DotDict()

config.config_Simulator = DotDict()

config.config_NeuralPlayer = DotDict()

config.config_NeuralPlayer.config_Datasets = DotDict()

config.config_NeuralPlayer.config_Datasets.ddqn = DotDict()
config.config_NeuralPlayer.config_Datasets.ddqn.sim = DotDict()

config.config_NeuralPlayer.config_Datasets.ae = DotDict()
config.config_NeuralPlayer.config_Datasets.ae.sim = DotDict()

config.config_NeuralPlayer.config_Agent = DotDict()
config.config_NeuralPlayer.config_Agent.config_Memory = DotDict()

config.config_NeuralPlayer.config_Preprocessing = DotDict()
config.config_NeuralPlayer.config_Preprocessing.config_AutoEncoder = DotDict()
config.config_NeuralPlayer.config_Preprocessing.config_AutoEncoder.config_AE_Datasets = DotDict()
config.config_NeuralPlayer.config_Preprocessing.config_AutoEncoder.config_AE_Datasets.config_S3 = DotDict()



config.config_HumanPlayer = DotDict()

config.config_HumanPlayer.config_Datasets = DotDict()
config.config_HumanPlayer.config_Datasets.sim = DotDict()


# -----------------------------------------------------------------
# General Info
# -----------------------------------------------------------------

config.min_steering = -5.0
config.max_steering = 5.0
config.min_throttle = 0.0
config.max_throttle = 1.0
config.action_space_boundaries = [[config.min_steering, config.max_steering], [config.min_throttle, config.max_throttle]]


# -----------------------------------------------------------------
# Simulator (the simulator launcher and gym env creator)
# -----------------------------------------------------------------
# This will be passed to gym.make after the port value is added to the dict

config.config_Simulator.update({"exe_path": "manual",
						"host": "127.0.0.1",
						"body_style": "donkey",
						"body_rgb": (128, 128, 128),
						"car_name": "me",
						"font_size": 100,
						"racer_name": "DDQN",
						"country": "FR",
						"bio": "Learning to drive w DDQN RL",
						"guid": str(uuid.uuid4()),
						"max_cte": 10,
				})



# -----------------------------------------------------------------
# Neural Player config
# -----------------------------------------------------------------

config_NeuralPlayer = config.config_NeuralPlayer

config_NeuralPlayer.agent_name               = "DQN"
config_NeuralPlayer.episodes                 = 3
config_NeuralPlayer.train_frequency          = 10
config_NeuralPlayer.camera_picture_shape     = (120, 160, 3)  # H * W * C
config_NeuralPlayer.cte_limit                = 3.2 # 3.2 is the white line
config_NeuralPlayer.cte_offset               = 2.25
config_NeuralPlayer.cte_coef                 = 1000 # cte goes from -3.2 to 3.2 on the road
config_NeuralPlayer.speed_coef               = 200 # speed goes aprox from 0 to 10
config_NeuralPlayer.reward_stick             = -1000
config_NeuralPlayer.replay_memory_freq       = 1
config_NeuralPlayer.replay_memory_batches    = 3

# -----------------------------------------------------------------
# Human Player config
# -----------------------------------------------------------------

config_HumanPlayer = config.config_HumanPlayer
config_HumanPlayer.min_steering = config.min_steering
config_HumanPlayer.max_steering = config.max_steering
config_HumanPlayer.min_throttle = config.min_throttle
config_HumanPlayer.max_throttle = config.max_throttle
config_HumanPlayer.coef				= 3
config_HumanPlayer.init_throttle	= 0.1
config_HumanPlayer.init_steering	= 0.1


# -----------------------------------------------------------------
# Datasets config
# -----------------------------------------------------------------

config_Datasets = config.config_NeuralPlayer.config_Datasets

config_Datasets.S3_connection			= True
if config_Datasets.S3_connection == True:
    config_Datasets.S3_bucket_name		= 'deyopotato'
    

# DDQN:
config_Datasets.ddqn = config.config_NeuralPlayer.config_Datasets.ddqn
config_Datasets.ddqn.load_model			= False
if config_Datasets.ddqn.load_model == True:
    config_Datasets.ddqn.load_name 		= "model_cache/ddqn/dedequene.modelo.1100" #if local: path from the root folder, if S3: path after bucket name
config_Datasets.ddqn.save_name 			= f"model_cache/ddqn/{config_NeuralPlayer.agent_name}_weights_{date}."
config_Datasets.ddqn.saving_frequency	= 0
config_Datasets.ddqn.save_score			= True


# SIMULATOR CACHE FOR DDQN:
config_Datasets.ddqn.sim = config.config_NeuralPlayer.config_Datasets.ddqn.sim
config_Datasets.ddqn.sim.load_name			= "simulator_cache/*"
config_Datasets.ddqn.sim.save				= False
if config_Datasets.ddqn.sim.save == True:
    config_Datasets.ddqn.sim.save_name		= f"simulator_cache/{config_NeuralPlayer.agent_name}_sim_{date}."
config_Datasets.ddqn.sim.size				= 3000


# AUTOENCODER:
config_Datasets.ae = config.config_NeuralPlayer.config_Datasets.ae
config_Datasets.ae.load_model			= True
if config_Datasets.ae.load_model == True:
    config_Datasets.ae.load_name 		= "model_cache/autoencoder/NiceAutoEncoder_h[8]_30K_examples" #if local: path from the root folder, if S3: path after bucket name
config_Datasets.ae.save_name			= "model_cache/autoencoder/dedes_autoencoder"
config_Datasets.ae.save_result			= False
if config_Datasets.ae.save_result == True:
	config_Datasets.ae.result_name		= "model_cache/autoencoder/images_results/dedes_autoencoder.png"


# SIMULATOR CACHE FOR AUTOENCODER:
config_Datasets.ae.sim = config.config_NeuralPlayer.config_Datasets.ae.sim
config_Datasets.ae.sim.load_name		= "simulator_cache/human_player/Human_sim_22_9.18_58"



# SIMULATOR CACHE FOR HUMAN PLAYER:
config.config_HumanPlayer.config_Datasets.S3_connection = config_Datasets.S3_connection
if config_Datasets.S3_connection == True:
	config.config_HumanPlayer.config_Datasets.S3_bucket_name = config_Datasets.S3_bucket_name
config.config_HumanPlayer.config_Datasets.sim.save_name		= f"simulator_cache/human_player/Human_sim_{date}"


# CONFIG:
config_Datasets.config_extension		= "config.json"


# -----------------------------------------------------------------
# Prepocessing
# -----------------------------------------------------------------

config_Preprocessing = config.config_NeuralPlayer.config_Preprocessing
config_Preprocessing.input_size         = config_NeuralPlayer.camera_picture_shape
config_Preprocessing.stack_size         = 4
config_Preprocessing.frame_skip         = 2  # interval in frames between the stacked frames
config_Preprocessing.shrink_size        = (120, 120) # * This does not remove the channels and generate a (60, 60) output. Channels are preserved :input (100, 100, 3) => (60, 60, 3)
config_Preprocessing.output_size        = (config_Preprocessing.stack_size, *config_Preprocessing.shrink_size) #*  C * H * W CHANNELS FIRST

config_Preprocessing.use_AutoEncoder	= True


# -----------------------------------------------------------------
# AutoEncoder
# -----------------------------------------------------------------

config_AutoEncoder = config.config_NeuralPlayer.config_Preprocessing.config_AutoEncoder

# Shapes
config_AutoEncoder.data					= config_NeuralPlayer.config_Datasets.ae
config_AutoEncoder.sim					= config_NeuralPlayer.config_Datasets.ae.sim
config_AutoEncoder.input_shape			= config_Preprocessing.output_size
config_AutoEncoder.bottleneck_size		= 8
config_AutoEncoder.layers_filters		= [3, 32, 32, 32, 64, 64, 128]

# Hyper Parameters
config_AutoEncoder.epochs				= 15
config_AutoEncoder.batch_size			= 64
config_AutoEncoder.lr					= 1e-3

config_AutoEncoder.show_plot			= False

if config_Preprocessing.use_AutoEncoder:
	config_Preprocessing.output_size    = (config_Preprocessing.stack_size, config_AutoEncoder.bottleneck_size)



agent_type = "DQN"
if (agent_type == "DQN"):
# -----------------------------------------------------------------
# Agent / training config
# -----------------------------------------------------------------

	config_Agent = config.config_NeuralPlayer.config_Agent

	config_Agent.agent_name         = "DDQN"
	config_Agent.input_size         = config_Preprocessing.output_size
	config_Agent.data               = config_NeuralPlayer.config_Datasets.ddqn
	config_Agent.sim                = config_NeuralPlayer.config_Datasets.ddqn.sim
	config_Agent.action_space_size  = (11, 3)
	config_Agent.discount           = 0.99
	config_Agent.lr                 = 1e-4
	config_Agent.initial_epsilon    = 0.9
	config_Agent.epsilon            = config_Agent.initial_epsilon
	config_Agent.epsilon_decay      = 0.9
	config_Agent.epsilon_min        = 0.02
	config_Agent.steps_to_eps_min   = 5000
	config_Agent.batch_size         = 2
	config_Agent.batches_number     = 10
	config_Agent.min_memory_size    = 258
	config_Agent.memory_size        = 1000
	config_Agent.num_workers        = 0 # set it to 0 if your computer can't handle multiprocessing
	
	config_Agent.target_model_update_frequency = 15
	config_Agent.action_space_boundaries =  config.action_space_boundaries



	config_Agent.action_space_boundaries   = config.action_space_boundaries
	action_space_length = 1
	
	tmp = []
	for i, size in enumerate(config_Agent.action_space_size):
		bounds = config.action_space_boundaries[i]
		tmp.append(np.linspace(start = bounds[0], stop = bounds[1], num = size))
	
	config_Agent.action_space = []
	for j in range(config_Agent.action_space_size[1]):
		for i in range(config_Agent.action_space_size[0]):
			config_Agent.action_space.append([tmp[0][i], tmp[1][j]])
	# print("config_Agent.action_space", config_Agent.action_space)
	

	if config_Preprocessing.use_AutoEncoder:
		config_Agent.with_AutoEncoder = True
	else:
		config_Agent.with_AutoEncoder = False
     

			
# -----------------------------------------------------------------
# Agent Memory config
# -----------------------------------------------------------------

	config_Memory = config.config_NeuralPlayer.config_Agent.config_Memory

	config_Memory.capacity = config_Agent.memory_size