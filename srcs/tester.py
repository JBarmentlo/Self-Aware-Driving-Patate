import argparse
from NeuralPlayer import NeuralPlayer

from SimulatorDummy import SimulatorDummy
from NeuralPlayerDummy import NeuralPlayerDummy
from configDummy import config
from torch.utils.data import DataLoader
import torch
import utils

config_NeuralPlayer = config.config_NeuralPlayer
config_Agent = config_NeuralPlayer.config_Agent


simulator = SimulatorDummy(config.config_Simulator,"donkey-generated-roads-v0")
def end():
	simulator.client.kill_sim()
	simulator.env.unwrapped.close()
	

neural = NeuralPlayerDummy(config.config_NeuralPlayer, env = simulator.env, simulator = simulator)
agent = neural.agent


while (len(agent.memory) <  config_Agent.batch_size):
	neural.do_races(1)

# batch_size = min(agent.config.batch_size, len(agent.memory))
# batch = agent.memory.sample(batch_size)
# train_dataloader = DataLoader(agent.memory, batch_size=batch_size, shuffle=True)
# loader_batch = next(iter(train_dataloader))

batch = agent.replay_memory()
s = batch[0]
ss = batch[2]
a = batch[1]
r = batch[3]
d = batch[4]
d = ~d
d = d.to(torch.int64)

qs = agent.model.forward(s)
qss = agent.target_model.forward(s)
qss_max, _  = torch.max(qss, dim = 1)

a_bin = utils.val_to_bin(a[0], [-3.0, 3.0], 7).to(torch.int64)
a2_bin = utils.val_to_bin(a[1], [1.0, 1.0], 1).to(torch.int64)
bin = a_bin * (a2_bin + 1)
# y.scatter(bin)
hot = torch.nn.functional.one_hot(bin, 7)
yj = hot * (r.view(7, -1) + qss_max.view(7, -1) * d.view(7, -1))
# target = hot * a_bin
# torch.reshape(r, (6, 1)) * hot


# _, actions_bin = torch.max(qs, axis = 1)



