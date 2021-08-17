import argparse
# from HumanPlayer import HumanPlayer
from NeuralPlayer import NeuralPlayer

from SimulatorDummy import SimulatorDummy
from NeuralPlayerDummy import NeuralPlayerDummy

from configDummy import config

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
	args = parse_arguments()

	if args.supervised:
		human = HumanPlayer(args)
	else:
		simulator = SimulatorDummy(config.config_Simulator, args.env_name)
		try:
			neural = NeuralPlayerDummy(config.config_NeuralPlayer, env = simulator.env, simulator=simulator)
			st = neural.env.reset()
			a = neural.agent.get_action(neural.preprocessor.process(st))
			print(a)

			neural.do_races(10)
		finally:
			simulator.client.kill_sim()
			simulator.env.unwrapped.close()
