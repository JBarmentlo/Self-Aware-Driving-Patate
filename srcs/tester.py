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

env_name = "donkey-generated-roads-v0"
simulator = Simulator(config.config_Simulator, env_name) 

neural = NeuralPlayer(config.config_NeuralPlayer, env = simulator.env, simulator=simulator)
# neural = NeuralPlayer(config.config_NeuralPlayer, None, None)
