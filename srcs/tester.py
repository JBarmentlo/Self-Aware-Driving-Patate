import argparse
from HumanPlayer import HumanPlayer
from NeuralPlayer import NeuralPlayer

from SimulatorDummy import SimulatorDummy
from NeuralPlayerDummy import NeuralPlayerDummy
from configDummy import config
from torch.utils.data import DataLoader



simulator = SimulatorDummy(config.config_Simulator,"donkey-generated-roads-v0")
def end():
	simulator.client.kill_sim()
	simulator.env.unwrapped.close()
	

neural = NeuralPlayerDummy(config.config_NeuralPlayer, env = simulator.env, simulator = simulator)
agent = neural.agent

# st = neural.env.reset()
# a = neural.agent.get_action(neural.preprocessor.process(st))
# while (len(agent.memory) <  64):
neural.do_races(10)

# batch_size = min(agent.config.batch_size, len(agent.memory))
# # batch = self.memory.sample(batch_size)
# train_dataloader = DataLoader(agent.memory, batch_size=batch_size, shuffle=True)
# batch = next(iter(train_dataloader))

