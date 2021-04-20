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
from utils import linear_bin
EPISODES = 20
img_rows, img_cols = 80, 80
# Convert image into Black and white

turn_bins = 7
img_channels = 4  # We stack 4 frames


def init_simulator(neural_player):
	# Sim config
	# only needed if TF==1.13.1
	neural_player.sim_config = tf.compat.v1.ConfigProto(log_device_placement=True)
	neural_player.sim_config.gpu_options.allow_growth = True
	print(neural_player.sim_config)

	# Keras session init
	neural_player.sess = tf.compat.v1.Session(config=neural_player.sim_config)
	K.set_session(neural_player.sess)

	# Create env
	neural_player.conf = {"exe_path": neural_player.args.sim,
					"host": "127.0.0.1",
					"port": neural_player.args.port,
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
	
	# Signal handler
	# not working on windows...
	def signal_handler(signal, frame):
			print("catching ctrl+c")
			neural_player.env.unwrapped.close()
			sys.exit(0)
	signal.signal(signal.SIGINT, signal_handler)
	signal.signal(signal.SIGTERM, signal_handler)
	signal.signal(signal.SIGABRT, signal_handler)

	return (neural_player)


class NeuralPlayer():
	def __init__(self, args):
		'''
		run a DDQN training session, or test it's result, with the donkey simulator
		'''
		self.args = args
		self = init_simulator(self)
		# Construct gym environment. Starts the simulator if path is given.
		self.env = gym.make(
			self.args.env_name, conf=self.conf)

		self.memory = deque(maxlen=10000)
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

	def prepare_state(self, obs, old_s_t=None):
		x_t = self.preprocessing.process_image(obs)
		if type(old_s_t) != type(np.ndarray):
			s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
			# In Keras, need to reshape
			s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  # 1*80*80*4
		else:
			x_t = x_t.reshape(1, x_t.shape[0], x_t.shape[1], 1)  # 1x80x80x1
			s_t = np.append(x_t, old_s_t[:, :, :, :3], axis=3)  # 1x80x80x4
		return s_t

	def reward_optimization(self, reward, done):
		if (done):
			reward = -1000
		return reward

	def run_ddqn(self):
		throttle = self.args.throttle  # Set throttle as constant value
		episodes = []
		for e in range(EPISODES):
			# Init
			print("Episode: ", e)
			done = False
			obs = self.env.reset()
			episode_len = 0

			# Apply preprocessing and stack 4 frames
			st = self.prepare_state(obs, old_s_t=None)

			while not done:

				# print(f"From env: cte {self.env.viewer.handler.cte}")
				# Choose action
				steering = self.agent.choose_action(st)
				# Get true value from categories
				action = [steering, throttle]
				# Do action
				next_obs, reward, done, info = self.env.step(action)
				# Reward opti
				reward = self.reward_optimization(reward, done)
				# Apply preprocessing and stack 4 frames
				st = self.prepare_state(obs, old_s_t=st)
				# Save the sample <s, a, r, s', d> to the memory
				self.memory.append((st,
                                    [np.argmax(linear_bin(action[0])), action[1]],
									reward,
                                    st,
									done))
				
				self.agent.update_epsilon()
				# if self.agent.t % 30 == 0:
				print(f"""Episode: {e}, Timestep: {self.agent.t}, Action: {action}, Reward: {reward}, Ep_len: {episode_len}, MaxQ: {self.agent.max_Q}""")
				self.agent.t = self.agent.t + 1
				episode_len = episode_len + 1
				if done:
					# Every episode update the target model to be same with model
					self.agent.update_target_model()
					episodes.append(e)
					# Save model for each episode
					if self.agent.train:
						self.agent.save_model(self.args.model)
					print("episode:", e, "  memory length:", len(self.agent.memory),
											"  epsilon:", self.agent.epsilon, " episode length:", episode_len)

			if self.agent.train:
				self.train_replay()

	def train_replay(self):
		# TODO: Bug in this function that need to be solved
		return None

		if len(self.memory) < self.agent.train_start:
			return
		batch_size = min(self.agent.batch_size, len(self.memory))
		minibatch = random.sample(self.memory, batch_size)
		state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
		state_t = np.concatenate(state_t)
		state_t1 = np.concatenate(state_t1)
		targets = self.agent.model.predict(state_t)
		self.agent.max_Q = np.max(targets[0])
		target_val = self.agent.model.predict(state_t1)
		target_val_ = self.agent.target_model.predict(state_t1)
		for i in range(batch_size):
			if terminal[i]:
				# BUG HERE: bad dereferencement -> shape is targets[][][].
				# Needs better understanding before solving isssue
				targets[i][action_t[i]] = reward_t[i]
			else:
				a = np.argmax(target_val[i])
				targets[i][action_t[i]] = reward_t[i] + \
					self.agent.discount_factor * (target_val_[i][a])
		self.agent.model.train_on_batch(state_t, targets)

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

