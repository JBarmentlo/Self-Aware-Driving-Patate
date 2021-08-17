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

# batch = agent.replay_memory()
# s = batch[0]
# ss = batch[2]
# a = batch[1]
# r = batch[3]
# d = batch[4]
# d = ~d
# d = d.to(torch.int64)

# qs = agent.model.forward(s)
# qss = agent.target_model.forward(s)
# qss_max, _  = torch.max(qss, dim = 1)

# a_bin = utils.val_to_bin(a[0], [-3.0, 3.0], 7).to(torch.int64)
# a2_bin = utils.val_to_bin(a[1], [1.0, 1.0], 1).to(torch.int64)
# bin = a_bin * (a2_bin + 1)
# # y.scatter(bin)
# hot = torch.nn.functional.one_hot(bin, 7)
# targets = qs.clone()
# targets.detach()
# targets = hot * (r.view(6, -1) + qss_max.view(6, -1) * d.view(6, -1)) - hot * qs + qs
# # target = hot * a_bin
# # torch.reshape(r, (6, 1)) * hot


# _, actions_bin = torch.max(qs, axis = 1)

agent.optimizer.zero_grad()


batch_size = min(agent.config.batch_size, len(agent.memory))
# batch = agent.memory.sample(batch_size)
train_dataloader = DataLoader(agent.memory, batch_size=batch_size, shuffle=True)
batch = next(iter(train_dataloader)) # s, a, s', r, d
s = batch[0].to(torch.float32)
ss = batch[2].to(torch.float32)
a = batch[1].to(torch.float32)
r = batch[3].to(torch.float32)
d = batch[4]
d = ~d
d = d.to(torch.int64)

qs = agent.model.forward(s).cpu()
qss = agent.target_model.forward(ss).cpu()
qss_max, _  = torch.max(qss, dim = 1)
action_space_size = [*agent.config.action_space_size]
a_bin = utils.val_to_bin(a[0], agent.config.action_space_boundaries[0], action_space_size[0]).to(torch.int64)
a2_bin = utils.val_to_bin(a[1], agent.config.action_space_boundaries[1], action_space_size[1]).to(torch.int64)
bin = a_bin * (a2_bin + 1)
# y.scatter(bin)
hot = torch.nn.functional.one_hot(bin, action_space_size[1] * action_space_size[0])
targets = qs.clone()
targets.detach()
targets = hot * (r.view(batch_size, -1) + agent.config.lr * (qss_max.view(batch_size, -1) * d.view(batch_size, -1))) - hot * qs + qs

error = agent.criterion(qs, targets).to(torch.float32)
error.backward()
agent.optimizer.step()

