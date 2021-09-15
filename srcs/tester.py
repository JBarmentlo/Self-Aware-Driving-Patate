from Simulator import Simulator
# from NeuralPlayer import NeuralPlayer
from HumanPlayer import HumanPlayer
from pynput import keyboard

from config import config

env_name = "donkey-generated-roads-v0"
simulator = Simulator(config.config_Simulator, env_name)

human = HumanPlayer(config.config_HumanPlayer, env = simulator.env, simulator = simulator)
human.do_race()