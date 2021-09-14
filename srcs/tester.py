import io
import torch
import pickle
import json

from Simulator import Simulator
from NeuralPlayer import NeuralPlayer
from ModelCache import ModelCache
from HumanPlayer import HumanPlayer
from S3 import S3
from inputs import get_key

from config import config

# env_name = "donkey-generated-roads-v0"
# simulator = Simulator(config.config_Simulator, env_name) 
#
# neural = NeuralPlayer(config.config_NeuralPlayer, env = simulator.env, simulator=simulator)
neural = NeuralPlayer(config.config_NeuralPlayer, None, None)

# neural.do_races(1)
with open("simulator_cache/DDQN_sim_27_8.12_28.3", "rb") as f:
    data = pickle.load(f)

state = data[0][0]
p_state = neural.preprocessor.process(state)
