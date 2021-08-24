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
import logging

ALogger = logging.getLogger("tester")
ALogger.setLevel(logging.INFO)
stream = logging.StreamHandler()
ALogger.addHandler(stream)

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
	neural.do_races(1)

simulator.client.release_sim()

ALogger.info(f"Replay from memory {len(agent.memory)}")

dataloader = DataLoader(agent.memory, batch_size=4,
                        shuffle=False, num_workers=1)

for i, single_batch in enumerate(dataloader):
	agent.update_target_model_counter += 1
	agent.optimizer.zero_grad()
	targets = []
	processed_states, actions, new_processed_states, rewards, dones = single_batch
	dones = ~dones

	qs_b = agent.model.forward(processed_states)
	qss_b = agent.target_model.forward(new_processed_states)
	qss_max_b, _ = torch.max(qss_b, dim = 1)

	for i, (action, reward, done) in enumerate(zip(actions, rewards, dones)):
		target = qs_b[i].clone()
		target = target.detach()
		a_idx = val_to_idx(action, agent.config.action_space)
		target[a_idx] = reward + (done * agent.config.discount * qss_max_b[i]) 
		targets.append(target)

	targets = torch.stack(targets)
	error = agent.criterion(qs_b, targets)
	error.backward()
	agent.optimizer.step()

# if (agent.update_target_model_counter % agent.config.target_model_update_frequency == 0):
# 	agent._update_target_model()

