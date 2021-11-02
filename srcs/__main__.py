import argparse

from Simulator import Simulator
from NeuralPlayer import NeuralPlayer
# from HumanPlayer import HumanPlayer

from config import config


import os
from itertools import count

import torch.multiprocessing as mp
import torch.distributed.rpc as rpc

from Players import CentralAgentWorker, CentralAgentMaster
from utils import free_all_sims


AGENT_NAME = "agent"
OBSERVER_NAME="worker{}"




def run_worker(rank, world_size):
	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = '29500'
	if rank == 0:
		# rank0 is the agent
		rpc.init_rpc(AGENT_NAME, rank=rank, world_size=world_size)

		Masta = CentralAgentMaster(config, world_size)
		EVAL_FREQUENCY = 1
		for i_episode in range(1000):
			# Masta.update_worker_agent_params()
			Masta.run_remote_episode(100)
			if ((i_episode % EVAL_FREQUENCY) == 0):
				Masta.run_eval_episode()		#TODO : Crashes in score.next function
				print("\n\n\nScores: ", Masta.scores)

		for woker_rref in Masta.worker_rrefs:
			woker_rref.rpc_sync().release_sim()
	else:
		# other ranks are the observer

		rpc.init_rpc(OBSERVER_NAME.format(rank), rank=rank, world_size=world_size)
		
	# block until all rpcs finish, and shutdown the RPC instance
	rpc.shutdown()






def parse_arguments():
	env_list = [
		"donkey-warehouse-v0",
		"donkey-generated-roads-v0",
		"donkey-avc-sparkfun-v0",
		"donkey-generated-track-v0",
		"donkey-roboracingleague-track-v0",
		"donkey-waveshare-v0",
		"donkey-minimonaco-track-v0",
		"donkey-warren-track-v0"
	]
	parser = argparse.ArgumentParser(description='ddqn')
	parser.add_argument('--sim', type=str, default="manual",
						help='path to unity simulator. maybe be left at manual if you would like to start the sim on your own.')
	parser.add_argument('--env_name', type=str, default="donkey-generated-roads-v0",
						help='name of donkey sim environment', choices=env_list)
	parser.add_argument('--agent', type=str, default="DQN",
						help='Choice of reinforcement Learning Agent (now determined by config file)', choices=["DQN", "SAC"])
	parser.add_argument('--no_sim', action='store_true',
						help='agent uses stored database to train')
	parser.add_argument('--supervised', action="store_true",
						help='Use Human Player instead of Neural Player')
	args = parser.parse_args()
	config.config_NeuralPlayer.agent_name = args.agent
	return (args)

if __name__ == "__main__":
	free_all_sims(config.num_workers)
	mp.spawn(
		run_worker,
		args=(config.num_workers + 1, ),
		nprocs=config.num_workers + 1,
		join=True
	)
		# simulator.env.unwrapped.close()


	# args = parse_arguments()
	# if args.no_sim == True:
	# 	neural = NeuralPlayer(config.config_NeuralPlayer, None, None)
	# 	neural.train_agent_from_SimCache()
	# else:
	# 	simulator = Simulator(config.config_Simulator, args.env_name)
	# 	try:
	# 		# if args.supervised == True:
	# 		# 	human = HumanPlayer(config.config_HumanPlayer, env = simulator.env, simulator = simulator)
	# 		# 	human.do_race()
	# 		# else:
	# 		neural = NeuralPlayer(config.config_NeuralPlayer, env = simulator.env, simulator=simulator)
	# 		neural.do_races(neural.config.episodes)
	# 	finally:
	# 		simulator.client.release_sim()
	# 		# simulator.env.unwrapped.close()
