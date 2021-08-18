import numpy as np
from utils_get_abs_path import get_path_to_cache
import uuid

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

config.config_NeuralPlayer.config_Preprocessing = DotDict()
config.config_NeuralPlayer.config_Preprocessing.config_AutoEncoder = DotDict()


# -----------------------------------------------------------------
# General Info
# -----------------------------------------------------------------

config.min_steering = -3.0
config.max_steering = 3.0
config.min_throttle = 1.0
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

config_NeuralPlayer.agent_name               = "random"
config_NeuralPlayer.episodes                 = 100
config_NeuralPlayer.train_frequency          = 10
config_NeuralPlayer.camera_picture_shape     = (120, 160, 3)  # H * W * C
config_NeuralPlayer.cte_limit                = 3.2
config_NeuralPlayer.cte_offset               = 2.25

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

    config_Agent.agent_name         = "DQN"
    config_Agent.input_size         = config_Preprocessing.output_size
    config_Agent.action_space_size  = (7, 1)
    config_Agent.discount           = 0.99
    config_Agent.lr                 = 1e-4
    config_Agent.initial_epsilon    = 1.0
    config_Agent.epsilon            = config_Agent.initial_epsilon
    config_Agent.epsilon_decay      = 0.9
    config_Agent.epsilon_min        = 0.02
    config_Agent.steps_to_eps_min   = 10000
    config_Agent.batch_size         = 64
    config_Agent.min_memory_size    = 100
    config_Agent.memory_size        = 10000



    config_Agent.action_space_boundaries   = config.action_space_boundaries
    config_Agent.action_space = [None] * len(config_Agent.action_space_size)

    for i, size in enumerate(config_Agent.action_space_size):
        bounds = config.action_space_boundaries[i]
        config_Agent.action_space[i] = np.linspace(start = bounds[0], stop = bounds[1], num = size)

# -----------------------------------------------------------------
# Agent Memory config
# -----------------------------------------------------------------

    config_Memory = config.config_NeuralPlayer.config_Agent.config_Memory

    config_Memory.capacity = config_Agent.memory_size
