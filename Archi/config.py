import numpy as np

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
config.sim_img_rows = 1000  # TODO: check real value
config.sim_img_cols = 1000  # TODO: check real value
# sim_img_channels is the number of colors in image
config.sim_img_channels = 3
config.sim_img_shape = (config.sim_img_rows,
                        config.sim_img_cols,
						config.sim_img_channels)

# Preprocessing
config.prep_img_rows = 80
config.prep_img_cols = 80
# prep_img_channels is the number of previous images given to the model
config.prep_img_channels = 4
config.prep_img_shape = (config.prep_img_rows,
                         config.prep_img_cols,
                         config.prep_img_channels)

# Model Prediction

# ----------------
# HYPER PARAMETERS
# ----------------



