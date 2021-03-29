'''
file: ddqn.py
author: Felix Yu
date: 2018-09-12
original: https://github.com/flyyufelix/donkey_rl/blob/master/donkey_rl/src/ddqn.py
'''
import os
import sys
import random
import argparse
import signal
import uuid
import numpy as np
import gym
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
from collections import deque
from preprocessing import Preprocessing
from agents.ddqn import DQNAgent
import gym_donkeycar
import tensorflow as tf
from tensorflow.compat.v1.keras import backend as K
EPISODES = 20
img_rows, img_cols = 80, 80
# Convert image into Black and white

turn_bins = 7
img_channels = 4  # We stack 4 frames


## Utils Functions ##
def linear_bin(a):
	"""
	Convert a value to a categorical array.
	Parameters
	----------
	a : int or float
		A value between -1 and 1
	Returns
	-------
	list of int
		A list of length 15 with one item set to 1, which represents the linear value, and all other items set to 0.
	"""
	a = a + 1
	b = round(a / (2 / (turn_bins - 1)))
	arr = np.zeros(turn_bins)
	arr[int(b)] = 1
	# print("bin", a, arr)
	return arr


def linear_unbin(arr):
	"""
	Convert a categorical array to value.
	See Also
	--------
	linear_bin
	"""
	if not len(arr) == turn_bins:
		raise ValueError('Illegal array length, must be 15')
	b = np.argmax(arr)
	a = b * (2 / (turn_bins - 1)) - 1
	# print("unbin", a, b)
	return a



class NeuralPlayer():
	def __init__(self, args):
		'''
		run a DDQN training session, or test it's result, with the donkey simulator
		'''
		self.args = args
		# only needed if TF==1.13.1
		self.sim_config = tf.compat.v1.ConfigProto(log_device_placement=True)
		self.sim_config.gpu_options.allow_growth = True
		print(self.sim_config)
		self.sess = tf.compat.v1.Session(config=self.sim_config)
		K.set_session(self.sess)
		self.conf = {"exe_path": args.sim,
                    "host": "127.0.0.1",
                    "port": args.port,
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
		# Construct gym environment. Starts the simulator if path is given.
		self.env = gym.make(args.env_name, conf=self.conf)
		# not working on windows...

		def signal_handler(signal, frame):
			print("catching ctrl+c")
			self.env.unwrapped.close()
			sys.exit(0)
		signal.signal(signal.SIGINT, signal_handler)
		signal.signal(signal.SIGTERM, signal_handler)
		signal.signal(signal.SIGABRT, signal_handler)
		# Get size of state and action from environment
		self.state_size = (img_rows, img_cols, img_channels)
		self.action_space = self.env.action_space  # Steering and Throttle
		self.agent = DQNAgent(self.state_size,
							  self.action_space,
							  input_shape=(img_rows, img_cols, img_channels),
							  output_size=turn_bins,
							  train=not args.test)
		self.preprocessing = Preprocessing()
		if os.path.exists(args.model):
			print("load the saved model")
			self.agent.load_model(args.model)
		try:
			self.run_ddqn()
		except KeyboardInterrupt:
			print("stopping run...")
		finally:
			self.env.unwrapped.close()

	def do_episode(self):
		pass

	def run_ddqn(self):
		throttle = self.args.throttle  # Set throttle as constant value
		episodes = []
		for e in range(EPISODES):
			print("Episode: ", e)
			done = False
			obs = self.env.reset()
			episode_len = 0
			x_t = self.preprocessing.process_image(obs)
			s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
			# In Keras, need to reshape
			s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  # 1*80*80*4
			while not done:
				print(f"From env: cte {self.env.viewer.handler.cte}")
				# Get action for the current state and go one step in environment
				steering = self.agent.get_action(s_t)
				action = [steering, throttle]
				next_obs, reward, done, info = self.env.step(action)
				# print(reward, done)
				if (done):
					reward = -1000
				x_t1 = self.preprocessing.process_image(next_obs)
				x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)  # 1x80x80x1
				s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)  # 1x80x80x4
				# Save the sample <s, a, r, s'> to the replay memory
				self.agent.replay_memory(s_t, np.argmax(
					linear_bin(steering)), reward, s_t1, done)
				self.agent.update_epsilon()
				if self.agent.train:
					self.agent.train_replay()
				s_t = s_t1
				self.agent.t = self.agent.t + 1
				episode_len = episode_len + 1
				if self.agent.t % 30 == 0:
					print("EPISODE",  e, "TIMESTEP", self.agent.t, "/ ACTION", action, "/ REWARD",
							reward, "/ EPISODE LENGTH", episode_len, "/ Q_MAX ", self.agent.max_Q)
				if done:
					# Every episode update the target model to be same with model
					self.agent.update_target_model()
					episodes.append(e)
					# Save model for each episode
					if self.agent.train:
						self.agent.save_model(self.args.model)
					print("episode:", e, "  memory length:", len(self.agent.memory),
											"  epsilon:", self.agent.epsilon, " episode length:", episode_len)


if __name__ == "__main__":
	# Initialize the donkey environment
	# where env_name one of:
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
	                    default="rl_driver.h5", help='path to model')
	parser.add_argument('--test', action="store_true",
	                    help='agent uses learned model to navigate env')
	parser.add_argument('--port', type=int, default=9091,
	                    help='port to use for websockets')
	parser.add_argument('--throttle', type=float, default=0.3,
	                    help='constant throttle for driving')
	parser.add_argument('--env_name', type=str, default="donkey-generated-roads-v0",
	                    help='name of donkey sim environment', choices=env_list)
	args = parser.parse_args()
	neural_player = NeuralPlayer(args)
	neural_player.run_ddqn()

