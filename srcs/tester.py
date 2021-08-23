import argparse

from Simulator import Simulator
from NeuralPlayer import NeuralPlayer
from config import config
from torch.utils.data import DataLoader
import torch
import utils
import numpy as np
from utils import val_to_idx
from S3 import S3
import json

config_NeuralPlayer = config.config_NeuralPlayer
config_Agent = config_NeuralPlayer.config_Agent
config_S3 = config_Agent.config_S3


simulator = Simulator(config.config_Simulator,"donkey-generated-roads-v0")
def end():
	simulator.client.kill_sim()
	simulator.env.unwrapped.close()
	

neural = NeuralPlayer(config.config_NeuralPlayer, env = simulator.env, simulator = simulator)
agent = neural.agent


while (len(agent.memory) <  config_Agent.batch_size):
	neural.do_races(10)

