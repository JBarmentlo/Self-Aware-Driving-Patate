import io
import torch
import pickle
import json

from Simulator import Simulator
from NeuralPlayer import NeuralPlayer
from HumanPlayer import HumanPlayer
from S3 import S3
from inputs import get_key

from config import config

# config_S3 = config.config_NeuralPlayer.config_Agent.config_Datasets.config_S3
# config_Agent = config.config_NeuralPlayer.config_Agent

# env_name = "donkey-generated-roads-v0"
# simulator = Simulator(config.config_Simulator, env_name) 

try:
    # human = HumanPlayer(config.config_NeuralPlayer, env = simulator.env, simulator=simulator)
    # neural = NeuralPlayer(config.config_NeuralPlayer, env = simulator.env, simulator=simulator)
    print("Getting keys:")
    i = 0
    while i < 100:
        event = get_key()[1]
        print(f"{event.state = }")
        i += 1
finally:
    # simulator.client.release_sim()
    pass
