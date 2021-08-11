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

# TODO: Ask for automated fin if os.environ["SIM_PATH"] doasnt exist (needed by server)

config = DotDict()

config.config_Simulator = DotDict()
config.config_NeuralPlayer = DotDict()
config.config_NeuralPlayer.config_Agent = DotDict()
config.config_NeuralPlayer.config_Preprocessing = DotDict()
config.config_NeuralPlayer.config_Agent.config_Memory = DotDict()



# -----------------------------------------------------------------
# Simulator
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
config_NeuralPlayer.camera_picture_shape     = (120, 160, 3)



# -----------------------------------------------------------------
# Prepocessing
# -----------------------------------------------------------------

config_Preprocessing = config.config_NeuralPlayer.config_Preprocessing

config_Preprocessing.input_size         = config_NeuralPlayer.camera_picture_shape
config_Preprocessing.output_size        = (120, 160, 3)




# -----------------------------------------------------------------
# Agent / training config
# -----------------------------------------------------------------

config_Agent = config.config_NeuralPlayer.config_Agent

config_Agent.agent_name         = "random"
config_Agent.input_size         = config_Preprocessing.output_size


# -----------------------------------------------------------------
# Agent Memory config
# -----------------------------------------------------------------

config_Memory = config.config_NeuralPlayer.config_Agent.config_Memory

config_Memory.size = 1
