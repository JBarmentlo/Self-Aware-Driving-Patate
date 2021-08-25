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

config.config_NeuralPlayer.config_Agent = DotDict()
config.config_NeuralPlayer.config_Agent.config_Memory = DotDict()
config.config_NeuralPlayer.config_Agent.config_S3 = DotDict()
config.config_NeuralPlayer.config_Agent.config_Database = DotDict()

config.config_NeuralPlayer.config_Preprocessing = DotDict()
config.config_NeuralPlayer.config_Preprocessing.config_AutoEncoder = DotDict()


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
config_NeuralPlayer.save_database            = True
config_NeuralPlayer.episodes                 = 2
config_NeuralPlayer.train_frequency          = 10
config_NeuralPlayer.camera_picture_shape     = (120, 160, 3)  # H * W * C
config_NeuralPlayer.cte_limit                = 4.0 # 3.2 is the white line
config_NeuralPlayer.cte_offset               = 2.25
config_NeuralPlayer.cte_coef                 = 1000 # cte goes from -3.2 to 3.2 on the road
config_NeuralPlayer.speed_coef               = 200 # speed goes aprox from 0 to 10
config_NeuralPlayer.reward_stick             = -1000
config_NeuralPlayer.replay_memory_freq       = 1
config_NeuralPlayer.replay_memory_batches    = 3




# -----------------------------------------------------------------
# Prepocessing
# -----------------------------------------------------------------

config_Preprocessing = config.config_NeuralPlayer.config_Preprocessing
config_Preprocessing.input_size         = config_NeuralPlayer.camera_picture_shape
config_Preprocessing.stack_size         = 4
config_Preprocessing.frame_skip         = 2  # interval in frames between the stacked frames
config_Preprocessing.shrink_size        = (80, 80) # * This does not remove the channels and generate a (60, 60) output. Channels are preserved :input (100, 100, 3) => (60, 60, 3)
config_Preprocessing.output_size        = (config_Preprocessing.stack_size, *config_Preprocessing.shrink_size) #*  C * H * W CHANNELS FIRST
config_Preprocessing.load_AutoEncoder	= True


# -----------------------------------------------------------------
# AutoEncoder
# -----------------------------------------------------------------

config_AutoEncoder = config.config_NeuralPlayer.config_Preprocessing.config_AutoEncoder


# Cache
config_AutoEncoder.model_dir			= "model_cache/autoencoder/"
config_AutoEncoder.train_dir			= "simulator_cache/"
config_AutoEncoder.test_dir				= "simulator_cache/"
config_AutoEncoder.name					= "Le_BG_du_13"

# Shapes
config_AutoEncoder.input_shape			= config_Preprocessing.output_size
config_AutoEncoder.output_shape			= 128

# Hyper Parameters
config_AutoEncoder.epochs				= 15
config_AutoEncoder.batch_size			= 64
config_AutoEncoder.lr					= 1e-3



agent_type = "DQN"
if (agent_type == "DQN"):
# -----------------------------------------------------------------
# Agent / training config
# -----------------------------------------------------------------

    config_Agent = config.config_NeuralPlayer.config_Agent

    config_Agent.agent_name         = "DDQN"
    config_Agent.input_size         = config_Preprocessing.output_size
    config_Agent.action_space_size  = (11, 3)
    config_Agent.discount           = 0.99
    config_Agent.lr                 = 1e-4
    config_Agent.initial_epsilon    = 0.9
    config_Agent.epsilon            = config_Agent.initial_epsilon
    config_Agent.epsilon_decay      = 0.9
    config_Agent.epsilon_min        = 0.02
    config_Agent.steps_to_eps_min   = 5000
    config_Agent.batch_size         = 128
    config_Agent.batches_number     = 10
    config_Agent.min_memory_size    = 256
    config_Agent.memory_size        = 10000
    config_Agent.load_model         = False
    config_Agent.model_to_load_path = "/workspaces/Self-Aware-Driving-Patate/model_cache/modelo/dedequene.modelo.2500"
    config_Agent.model_to_save_path = f"/workspaces/Self-Aware-Driving-Patate/model_cache/modelo/{config_Agent.agent_name}_{date}." ### TODO : improve path
    config_Agent.saving_frequency = 100
    config_Agent.target_model_update_frequency = 15
    config_Agent.action_space_boundaries =  config.action_space_boundaries



    config_Agent.action_space_boundaries   = config.action_space_boundaries
    action_space_length = 1
    
    tmp = []
    for i, size in enumerate(config_Agent.action_space_size):
        bounds = config.action_space_boundaries[i]
        tmp.append(np.linspace(start = bounds[0], stop = bounds[1], num = size))
    
    print("tmp", tmp)
    config_Agent.action_space = []
    for j in range(config_Agent.action_space_size[1]):
        for i in range(config_Agent.action_space_size[0]):
            config_Agent.action_space.append([tmp[0][i], tmp[1][j]])
    print("config_Agent.action_space", config_Agent.action_space)
    

            

# -----------------------------------------------------------------
# Agent Memory config
# -----------------------------------------------------------------

    config_Memory = config.config_NeuralPlayer.config_Agent.config_Memory

    config_Memory.capacity = config_Agent.memory_size



# -----------------------------------------------------------------
# Player Database config
# -----------------------------------------------------------------

    config_Database = config.config_NeuralPlayer.config_Agent.config_Database

    config_Database.on                  = True
    config_Database.max_datapoints      = 10000
    config_Database.local_model_path          = f"/workspaces/Self-Aware-Driving-Patate/simulator_cache/ddqn_track/{config_Agent.agent_name}_{date}."




# -----------------------------------------------------------------
# S3 config
# -----------------------------------------------------------------

    config_S3 = config.config_NeuralPlayer.config_Agent.config_S3

    config_S3.bucket_name = 'deyopotato'
    config_S3.upload = True ### True if you want to upload your models locally and on s3
    config_S3.download = False ### True if you need to download a model file from s3
    config_S3.s3_model_path = 'model_cache/dedequene.modelo.1100' ### s3 file path inside deyopotato 
    config_S3.s3_model_folder = 'model_cache/'
    config_S3.local_model_path = '/workspaces/Self-Aware-Driving-Patate/model_cache/modelo/from_s3_modelo'
    config_S3.s3_sim_path = f'sim_cache/ddqn_{config_Agent.agent_name}_{date}.'
    config_S3.s3_sim_folder = 'sim_cache/'
