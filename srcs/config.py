import numpy as np
from utils_get_abs_path import get_path_to_cache

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

config = DotDict()

# --------------------
# INPUT DATA DIMENSION
# --------------------

# Simulator
config.EPISODES = 10_000
config.img_rows, config.img_cols = 64, 64
config.turn_bins = 7
config.img_channels = 4

config.sim_img_rows = 120  # TODO: check real value : OK Gilles
config.sim_img_cols = 160  # TODO: check real value : OK Gilles
# sim_img_channels is the number of colors in image
config.sim_img_channels = 3
config.sim_img_shape = (config.sim_img_rows,
                        config.sim_img_cols,
						config.sim_img_channels)

# Preprocessing
config.prep_img_rows = 64
config.prep_img_cols = 64
# prep_img_channels is the number of previous images given to the model
config.prep_img_channels = 4
config.prep_img_shape = (config.prep_img_rows,
                         config.prep_img_cols,
                         config.prep_img_channels)
# size of the vector at the encoder output
config.output_shape = 128 
config.encoder_output_shape = 128 

# Model Prediction
config.turn_bins = 7

cte_config = DotDict()
cte_config.cte_offset = 2.25
cte_config.max_cte = 3.2
cte_config.done_func = lambda x: abs(x) > cte_config.max_cte

# ----------------
# HYPER PARAMETERS
# ----------------
config.init_throttle = 0.5
config.init_steering = 0.3
config.min_throttle, config.max_throttle = -1, 1
config.min_steering, config.max_steering = -1, 1 
config.max_threads = 16
config.coef = 1.7

config.name_neural_player = "NeuralPlayer"
config.name_human_player = "HumanPlayer"
config.local_memory_folder = "local_memory"
config.s3_memory_folder = "memory"
config.memory_sufix = ".pkl"
config.info_sufix = "_infos.json"
config.main_folder = get_path_to_cache("")
config.bucket_name = "deyopotato"

# ----------------------------
# Hyperparameters AutoEncoder
# ----------------------------
config.epochs=100
config.batch_size=128

