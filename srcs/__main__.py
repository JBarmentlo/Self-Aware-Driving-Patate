import argparse

from Simulator import Simulator
from NeuralPlayer import NeuralPlayer
from HumanPlayer import HumanPlayer

from config import config

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
	args = parse_arguments()
	if args.no_sim == True:
		neural = NeuralPlayer(config.config_NeuralPlayer, None, None)
		neural.train_agent_from_SimCache()
	else:
		simulator = Simulator(config.config_Simulator, args.env_name)
		try:
			if args.supervised == True:
				human = HumanPlayer(config.config_HumanPlayer, env = simulator.env, simulator = simulator)
				human.do_race()
			else:
				neural = NeuralPlayer(config.config_NeuralPlayer, env = simulator.env, simulator=simulator)
				neural.do_races(neural.config.episodes)
		finally:
			simulator.client.release_sim()
			# simulator.env.unwrapped.close()
