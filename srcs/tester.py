import argparse
from NeuralPlayer import NeuralPlayer

from SimulatorDummy import SimulatorDummy
from NeuralPlayerDummy import NeuralPlayerDummy
from configDummy import config
from torch.utils.data import DataLoader
import torch
import utils


simulator = SimulatorDummy(config.config_Simulator,"donkey-generated-roads-v0")
def end():
	simulator.client.kill_sim()
	simulator.env.unwrapped.close()
	

neural = NeuralPlayerDummy(config.config_NeuralPlayer, env = simulator.env, simulator = simulator)
agent = neural.agent


# while (len(agent.memory) <  100):
# 	neural.do_races(1)

# # batch_size = min(agent.config.batch_size, len(agent.memory))
# # batch = agent.memory.sample(batch_size)
# # train_dataloader = DataLoader(agent.memory, batch_size=batch_size, shuffle=True)
# # loader_batch = next(iter(train_dataloader))

# batch = agent.replay_memory()
# s = batch[0]
# ss = batch[2]
# a = batch[1]
# r = batch[3]
# d = batch[4]
# qs = agent.model.forward(s)
# _, actions_bin = torch.max(qs, axis = 1)



