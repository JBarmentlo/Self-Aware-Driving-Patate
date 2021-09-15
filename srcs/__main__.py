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

        for i_episode in range(3):
            Masta.update_worker_agent_params()
            Masta.run_remote_episode(1000)

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
    parser.add_argument('--model', type=str,
                        default="rl_driver", help='path to model')
    parser.add_argument('--test', action="store_true",
                        help='agent uses learned model to navigate env')
    parser.add_argument('--port', type=int, default=9091,
                        help='port to use for websockets')
    parser.add_argument('--throttle', type=float, default=0.3,
                        help='constant throttle for driving')
    parser.add_argument('--env_name', type=str, default="donkey-generated-roads-v0",
                        help='name of donkey sim environment', choices=env_list)
    parser.add_argument('--agent', type=str, default="DDQN",
                        help='Choice of reinforcement Learning Agent (now determined by config file)', choices=["DDQN", "SAC"])
    parser.add_argument('--no_sim', type=str, default=False,
                        help='agent uses stored database to train')
    parser.add_argument('--save', action="store_true",
                        help='Saving each episode in a pickle file')
    parser.add_argument('--destination', default="local",
                        help='Choice of destination to save the memory', choices=["local", "s3"])
    parser.add_argument('--supervised', action="store_true",
                        help='Use Human Player instead of Neural Player')
    args = parser.parse_args()
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
