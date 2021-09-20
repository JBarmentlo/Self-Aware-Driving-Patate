from Simulator import Simulator
from NeuralPlayer import NeuralPlayer
# from HumanPlayer import HumanPlayer
# from pynput import keyboard

from config import config

# env_name = "donkey-generated-roads-v0"
# simulator = Simulator(config.config_Simulator, env_name)

neural = NeuralPlayer(config.config_NeuralPlayer, None, None)
neural.train_agent_from_SimCache()