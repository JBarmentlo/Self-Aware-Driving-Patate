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

turn_bins = 7 ### TODO config.py
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
		self.action_space = self.env.action_space  # Steering and Throttle ### TODO
		self.agent = DQNAgent(self.state_size,
							  self.action_space,
							  input_shape=(img_rows, img_cols, img_channels),
							  output_size=turn_bins,
							  train=not args.test)
		self.preprocessing = Preprocessing()

		# For numpy print formating:
		np.set_printoptions(precision=4)

		if os.path.exists(args.model):
			print("load the saved model")
			self.agent.load_model(args.model)
		try:
			self.run_ddqn()
		except KeyboardInterrupt:
			print("stopping run...")
		finally:
			self.env.unwrapped.close()

	def prepare_state(self, state, old_state=None): ### TODO: rename old state
		x_t = self.preprocessing.process_image(state)
		if type(old_state) != type(np.ndarray):
			# For 1st iteration when we do not have old_state
			s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
			# In Keras, need to reshape
			s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  # 1*80*80*4
		else:
			x_t = x_t.reshape(1, x_t.shape[0], x_t.shape[1], 1)  # 1x80x80x1
			s_t = np.append(x_t, old_state[:, :, :, :3], axis=3)  # 1x80x80x4 
		return s_t

	def reward_optimization(self, reward, done):
		if (done):
			reward = -1000.0
		return reward

	def run_ddqn(self):
		throttle = self.args.throttle  # Set throttle as constant value
		episodes = []
		for e in range(EPISODES):
			# Init
			print("Episode: ", e)
			terminal = False ## TODO : maybe done ?
			state = self.env.reset()
			episode_len = 0

			# Apply preprocessing and stack 4 frames
			preprcocessed_state = self.prepare_state(state, old_state=None)

			while not terminal:

				# print(f"From env: cte {self.env.viewer.handler.cte}")
				# Choose action
				# TODO: It is time to make the model decide the throttle itself
				steering = self.agent.choose_action(preprcocessed_state)
				# Adding throttle
				action = [steering, throttle]
				# Do action
				new_state, reward, terminal, info = self.env.step(action)
				# Reward opti
				reward = self.reward_optimization(reward, terminal)
				# Apply preprocessing and stack 4 frames
				new_preprcocessed_state = self.prepare_state(new_state, old_state=state)

				# MEMORY for Train_replay():
				# Save the sample <s, a, r, s', d> to the memory
				# 	state:		Is the state directly usable by agent, after preprocessing
				# 	action:		Direct values [steering, throttle] for interaction with simulators (floats between [-1, 1] and [0, 1])
				#	reward:		Reward for action.
				#	new_state:	New state resulting from 'action'
				# 	terminal:	Is at True when game is over 
				self.memory.append((preprcocessed_state,
                                    action,
									reward,
                                    new_preprcocessed_state,
									terminal)) ### TODO: info?
				
				self.agent.update_epsilon()
				# if self.agent.t % 30 == 0:
				print(
					f"""Episode: {e}, t: {self.agent.t:<5} Action: [{action[0]:6.3} {action[1]:6.3}], Reward: {reward:6.4} Ep_len: {episode_len:<5} MaxQ: {self.agent.max_Q:3.3}""")
				self.agent.t = self.agent.t + 1
				episode_len = episode_len + 1 ### TODO: check redondance
				if terminal:
					# Every episode update the target model to be same with model
					self.agent.update_target_model()
					episodes.append(e) ### TODO : why?
					# Save model for each episode
					if self.agent.train:
						self.agent.save_model(self.args.model) ### TODO: faire un truc propre avec os
					print(f"episode: {e} memory length: {len(self.agent.memory)} epsilon: {self.agent.epsilon} episode length: {episode_len}")
				
				# Updating state variables
				state = new_state
				preprcocessed_state = new_preprcocessed_state

			if self.agent.train:
				self.train_replay()

	def train_replay(self):
		if len(self.memory) < self.agent.train_start:
			return
		print(f"Train replay on {len(self.memory)} elements")
		batch_size = min(self.agent.batch_size, len(self.memory))
		minibatch = random.sample(self.memory, batch_size)
		# For data structure look for comment in run_ddqn() 
		state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch) ### TODO: add info
		state_t = np.concatenate(state_t) ### TODO: et dans le preprocessing?
		state_t1 = np.concatenate(state_t1)
		# Targets, are the predictions from agent.
		# Currently (april 20) they are 7 categories corresponding to values of steering
		# The agent predicts Q-Values for each of these categories
		targets = self.agent.model.predict(state_t)
		# Use of agent.max_Q is for printing
		self.agent.max_Q = np.max(targets)
		# We predict new state to be able to update Q-Values
		target_val_predict = self.agent.model.predict(state_t1)
		target_val_update = self.agent.target_model.predict(state_t1)
		for i in range(batch_size):
			# Here: we convert action_t which is a float directly used by simulator, to a category, which is what the model currently predicts
			# The last 0 is the angle, as for the moment we are not interested in the throttle
			# we use linear_bin, to convert float to categories
			bin_action = np.argmax(linear_bin(action_t[i][0]))
			if terminal[i]:
				targets[i][bin_action] = reward_t[i]
			else:
				# We get the most recent prediction from agent of the new_state obtained from action_t 
				a = np.argmax(target_val_predict[i])
				targets[i][bin_action] = reward_t[i] + \
					self.agent.discount_factor * (target_val_update[i][a])
		# Now that all the targets have been updated, we can retrain the agent
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

