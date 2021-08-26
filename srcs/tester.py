import io
import torch
import pickle

from Simulator import Simulator
from NeuralPlayer import NeuralPlayer
from HumanPlayer import HumanPlayer
from S3 import S3

from config import config

config_S3 = config.config_NeuralPlayer.config_Agent.config_S3
config_Agent = config.config_NeuralPlayer.config_Agent

neural = NeuralPlayer(config.config_NeuralPlayer, None, None)
agent = neural.agent

# config_Agent.S3_connection = True
name = "model_cache/dedequene.modelo.2500"
name2= "model_cache/weshwesh2500"
