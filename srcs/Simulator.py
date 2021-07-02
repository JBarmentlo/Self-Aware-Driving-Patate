import signal

# TODO: FOLLOWING LINES ARE AN IMPERFECT FIX 
# TODO -> SHOULD BE CHANGED IF BETTER SOLUTION CAN BE THOUGHT
# 	Great resource: https://help.pythonanywhere.com/pages/DebuggingImportError/
#
# 	It's hard to correctly import simlaunch3000
#	because when python is importing files it uses sys.path
#	to find the file, this depends on the original argument
#	when launching the project
#
#	So, why not directly changes the way we import in simlaunch3000 ?
#	We are using a submodule for the simlaunch3000 as it needs to be 
#	used in 2 different ways, which neither starts with the same sys.path[0] 		
#		- as an independent program outside Docker 
#		- as a call to the client class in Docker
#	We dio not want to get rid of the submodule as it will become 
#	harder to maintain in the long term 
#	And, we would like to encourage to other parts of this project
#	to become submodules
#
#	Solution ?
#		Why not creating packages which we could import ?
import sys
from utils_get_abs_path import get_path_to_cache
path = get_path_to_cache("src/simlaunch3000/src/")
if path not in sys.path:
    sys.path.insert(0, path)
from simlaunch3000 import Client
# TODO: end of todo

import tensorflow as tf
import gym
import uuid
import gym_donkeycar ## Keep this module 
from utils import save_memory_db
import sys
from tensorflow.compat.v1.keras import backend as K
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Simulator:
	def __init__(self, player):
		# Sim config
		# only needed if TF==1.13.1
		player.sim_config = tf.compat.v1.ConfigProto(log_device_placement=True)
		player.sim_config.gpu_options.allow_growth = True
		print(player.sim_config)

		# Keras session init
		player.sess = tf.compat.v1.Session(config=player.sim_config)
		K.set_session(player.sess)


		if player.args.sim == "simlaunch3000":
			exe_path = "manual"
			player.client = Client()
			player.client.request_simulator()
			player.args.port = player.client.sim_port
			player.args.port = 9093
		else:
			exe_path = player.args.sim
		# Create env
		player.conf = {"exe_path": exe_path,
						"host": "127.0.0.1",
						"port": player.args.port,
						"body_style": "donkey",
						"body_rgb": (128, 128, 128),
						"car_name": "me",
						"font_size": 100,
						"racer_name": "DDQN",
						"country": "FR",
						"bio": "Learning to drive w DDQN RL",
						"guid": str(uuid.uuid4()),
						"max_cte": 10,
				}
		print(player.conf)
		player.env = gym.make(
				player.args.env_name, conf=player.conf)
		# Signal handler
		# not working on windows...
		def signal_handler(signal, frame):
				print("catching ctrl+c")
				if player.args.save or player.args.supervised:
					save_memory_db(player.episode_memory, player.general_infos, "last")
				player.env.unwrapped.close()
				sys.exit(0)
		signal.signal(signal.SIGINT, signal_handler)
		signal.signal(signal.SIGTERM, signal_handler)
		signal.signal(signal.SIGABRT, signal_handler)
