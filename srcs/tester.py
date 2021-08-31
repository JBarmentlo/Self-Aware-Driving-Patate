import io
import torch
import pickle

from Simulator import Simulator
from NeuralPlayer import NeuralPlayer
from HumanPlayer import HumanPlayer
from S3 import S3

from config import config

config_S3 = config.config_NeuralPlayer.config_Agent.config_Datasets.config_S3
config_Agent = config.config_NeuralPlayer.config_Agent

env_name = "donkey-generated-roads-v0"
simulator = Simulator(config.config_Simulator, env_name) 


# neural = NeuralPlayer(config.config_NeuralPlayer, None, None)
neural = NeuralPlayer(config.config_NeuralPlayer, env = simulator.env, simulator=simulator)
agent = neural.agent
while(len(agent.memory) < config_Agent.min_memory_size):
    neural.do_races(1)


