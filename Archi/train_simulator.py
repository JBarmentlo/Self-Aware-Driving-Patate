import os
import sys
import random
import argparse
import signal
import uuid
import numpy as np
import gym
# import skimage as skimage
# from skimage import transform, color, exposure
# from skimage.transform import rotate
# from skimage.viewer import ImageViewer
from collections import deque
from preprocessing import Preprocessing
from agents.ddqn import DQNAgent
from agents.sac import SoftActorCritic
import gym_donkeycar
import tensorflow as tf
from tensorflow.compat.v1.keras import backend as K
from utils import is_cte_out
### Deya
import pickle
from gym.spaces import Box
from inputs import get_key
from config import config
import threading
from datetime import datetime
import json
import boto3
import os
### UNCONMMENT THE FOLLOWING LINE TO CONNECT TO S3 BUCKET:
# from s3 import s3, bucket_name, bucket

# doesn't show TF warnings..
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def upload_pickle_file(file_name, content):
	### UNCOMMENT THESE LINES IF YOU WANT TO UPLOAD DIRECTLY TO S3:
	# pickle_byte_obj = pickle.dumps(content)
	# s3.Object(bucket_name, file_name).put(Body=pickle_byte_obj)
	
	### COMMENT THESE LINES IF YOU DON't WANT TO UPLOAD THE PKL FILE LOCALLY:
	with open(file_name, "wb") as f:
		pickle.dump(content, f)


def read_s3_pkl_file(name):
	### UNCOMMENT THESE LINES IF YOU WANT TO READ THE FILE DIRECTLY FROM S3:
	# result = pickle.loads(bucket.Object(name).get()['Body'].read())
	
	### COMMENT THESE LINES IF YOU DON't WANT TO READ THE FILE LOCALLY:
	with open(name, "rb") as f:
		result = json.load(f)
	return(result)


def upload_json_file(file_name, content):
	### UNCOMMENT THESE LINES IF YOU WANT TO UPLOAD THE FILE DIRECTLY TO S3:
	# json_byte_obj = json.dumps(content)
	# s3.Object(bucket_name, file_name).put(Body=json_byte_obj)
	
	### COMMENT THESE LINES IF YOU DON't WANT TO UPLOAD THE JSON FILE LOCALLY:
	with open(file_name, "w") as f:
		json.dump(content, f)
	

def init_simulator(player):
	# Sim config
	# only needed if TF==1.13.1
	player.sim_config = tf.compat.v1.ConfigProto(log_device_placement=True)
	player.sim_config.gpu_options.allow_growth = True
	print(player.sim_config)

	# Keras session init
	player.sess = tf.compat.v1.Session(config=player.sim_config)
	K.set_session(player.sess)

	# Create env
	player.conf = {"exe_path": player.args.sim,
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

	return (player)


def append_db(episode_memory, state, action, reward, new_state, done, info):
	### TODO: rajouter check des arguments
	# MEMORY for database (to train without simulator)
	episode_memory.append((state, action, reward, new_state, done, info))
	

def save_memory_db(memory_list, infos, episode):
	file_name = f"{infos['prefix']}_{episode}{config.memory_sufix}"
	upload_pickle_file(file_name, memory_list)


def init_dic_info(args): ### TODO add infos about last commit ec...
	date = datetime.now().strftime("%d_%m_%Hh%Mm")
	name = config.name_neural_player
	if args.supervised:
		name = config.name_human_player
	info_prefix = f"{config.memory_folder}/{name}_{date}"
	infos = {"name" : name, "date" : str(date), "env_name" : args.env_name, "prefix" : info_prefix}
	info_file_name = f"{info_prefix}{config.info_sufix}"
	upload_json_file(info_file_name, infos)
	return (infos)

class HumanPlayer():
	def __init__(self, args):
		self.args = args
		self = init_simulator(self)
		self.episode_memory = []
		self.db = None
		self.db_len = 0
		self.general_infos = init_dic_info(self.args)
		self.episode_memory = []
		self.throttle, self.steering, self.stop = 0, 0, 0
		self.commands = self.throttle, self.steering, self.stop
		try:
			self.run_supervised()
		except KeyboardInterrupt:
			print("stopping run...")
		finally:
			self.env.unwrapped.close()
	
	def run_supervised(self):
		print("-------- PRESS any key to start connecting the keyboard, it can take a while...")
		state = self.env.reset()
		get_key()
		print("\n\n**********         Now you can start driving with your KEYPADS :) :)         **********\n\n")
		while self.stop == 0:
			commands = self.commands
			self.get_command()
			if self.commands != commands:
				action = [self.steering, self.throttle]
				new_state, reward, done, info = self.env.step(action)
			else:
				action = None
				new_state, reward, done, info = self.env.viewer.observe()
			if threading.active_count() <= config.max_threads:
				t = threading.Thread(target=append_db, args=[self.episode_memory, state, action, reward, new_state, done, info])
				t.start()
			state = new_state
		print("stopping")
		save_memory_db(self.episode_memory, self.general_infos, 0)
		
	
	def get_command(self):
		event = get_key()[1]
		if event.code == "KEY_ESC" and event.state == 1:
			self.stop = 1
		elif event.code == "KEY_UP" and event.state == 1:
			self.throttle = abs(self.throttle * config.coef)
		elif (event.code == "KEY_UP" or event.code == "KEY_DOWN") and event.state == 0:
			self.throttle = config.init_throttle
		elif event.code == "KEY_DOWN" and event.state == 1:
			self.throttle = abs(self.throttle * config.coef) * -1
		elif event.code == "KEY_LEFT" and event.state == 1:
			if self.steering == 0:
				self.steering = config.init_steering * -1
			else:
				self.steering = abs(self.steering * config.coef) * -1
		elif (event.code == "KEY_LEFT" or event.code == "KEY_RIGHT") and event.state == 0:
			self.steering = 0
		elif event.code == "KEY_RIGHT" and event.state == 1:
			if self.steering == 0:
				self.steering = config.init_steering
			else:
				self.steering = abs(self.steering * config.coef)
		self.check_max_min()
		self.commands = self.stop, self.throttle, self.steering 
		
	def check_max_min(self):
		if self.throttle > config.max_throttle:
			self.throttle = config.max_throttle
		if self.throttle < config.min_throttle:
			self.throttle = config.min_throttle
		if self.steering > config.max_steering:
			self.steering = config.max_steering
		if self.steering < config.min_steering:
			self.steering = config.min_steering 


class NeuralPlayer():
	def __init__(self, args):
		'''
		run a DDQN training session, or test it's result, with the donkey simulator
		'''
		self.args = args
		if not self.args.no_sim:
			self = init_simulator(self)
		# Construct gym environment. Starts the simulator if path is given.
		self.memory = deque(maxlen=10000)
		self.episode_memory = []
		self.db = None
		self.db_len = 0
		if self.args.save:
			self.general_infos = init_dic_info(self.args)
		# Get size of state and action from environment
		self.state_size = (config.img_rows, config.img_cols, config.img_channels)
		self.action_space = Box(-1.0, 1.0, (2,), dtype=np.float32) ### TODO: not the best
		# self.action_space = self.env.action_space  # Steering and Throttle
		if args.agent == "DDQN": 
			self.agent = DQNAgent(self.state_size,
								self.action_space,
								input_shape=(config.img_rows, config.img_cols, config.img_channels),
								output_size=config.turn_bins,
								train=not args.test)
		elif args.agent == "SAC":
			self.agent = SoftActorCritic(self.state_size,
								self.action_space,
								input_shape=(config.img_rows, config.img_cols, config.img_channels),
								output_size=config.turn_bins,
								learning_rate=1e-4,
								train=not args.test)
		self.preprocessing = Preprocessing()

		# For numpy print formating:
		np.set_printoptions(precision=4)

		if os.path.exists(args.model):
			print("load the saved model")
			# TODO: Carefull, when we will have 2 different agents available (DDQN & SAC),
			# TODO: 	it will be an easy mistake to load the wrong one.
			# TODO: 	We need to protect against it
			self.agent.load_model(args.model)
		try:
			self.run_agent()
		except KeyboardInterrupt:
			print("stopping run...")
		finally:
			if not self.args.no_sim:
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
			# TODO: Carefull with reward if terminal state is after winning the race -> should be positive
			reward = -1000.0
		return reward

	def get_db_from_file(self, name, episode):
		file_name = f"{name}_{episode}{config.memory_sufix}"
		if os.path.exists(file_name):
			self.db = read_pickle_file(file_name)
			self.db_len = len(self.db)
		else:
			return (False)
		return (True)
	
	def save_memory_train(self, preprocessed_state, action, reward, new_preprocessed_state, done, info):
		### TODO: rajouter check des arguments
		# MEMORY for Train_replay():
				# Save the sample <s, a, r, s', d> to the memory
				# 	preprocessed_state:		Is the state directly usable by agent, after preprocessing
				# 	action:		            Direct values [steering, throttle] for interaction with simulators (floats between [-1, 1] and [0, 1])
				#	reward:		            Reward for action.
				#	new_preprocessed_state:	New state resulting from 'action'
				# 	done:	                Is at True when game is over
				#   info:                   info about velocity, cte ... etc
		self.memory.append((preprocessed_state, action, reward, new_preprocessed_state, done, info))
	
	def run_agent(self):
		for e in range(config.EPISODES):
			print("Episode: ", e)
			episode_len = 0
			if self.args.no_sim != False:
				res = self.get_db_from_file(self.args.no_sim, e)
				if res == False:
					break
				state, _, _, _, done, _ = self.db[episode_len]
			else:
				done = False
				state = self.env.reset()
				throttle = self.args.throttle  # Set throttle as constant value
			print(f"done = {done}")
			while not done:
				# Apply preprocessing and stack 4 frames
				preprocessed_state = self.prepare_state(state, old_state=None)
				# print(f"From env: cte {self.env.viewer.handler.cte}")
				# Choose action
				# TODO: It is time to make the model decide the throttle itself
				if not self.args.no_sim:
					steering = self.agent.choose_action(preprocessed_state)
					# Adding throttle
					action = [steering, throttle]
					# Do action
					new_state, reward, done, info = self.env.step(action)
					# episode_len > 10 because sometimes
					# 	simulator gives cte value from previous episode at the begining
					if episode_len > 10 and is_cte_out(info['cte']):
						done = True
					if done == True:
						print("doonnnnnnnnnnnnne*************")
				else:
					_, action, reward, new_state, done, info = self.db[episode_len]
				# Reward opti
				reward = self.reward_optimization(reward, done)
				# Apply preprocessing and stack 4 frames
				new_preprocessed_state = self.prepare_state(new_state, old_state=state)
				
				self.save_memory_train(preprocessed_state, action, reward, new_preprocessed_state, done, info)
				
				if self.args.save:
					append_db(self.episode_memory, state, action, reward, new_state, done, info)
				
				self.agent.update_epsilon()
				# if self.agent.t % 30 == 0:
				# print(f"Episode: {e}, episode_len: {episode_len:<5} Action: [{action[0]:6.3} {action[1]:6.3}], Reward: {reward:6.4} Ep_len: {episode_len:<5} MaxQ: {self.agent.max_Q:3.3}")
				episode_len = episode_len + 1
				if done or (self.db_len != 0 and episode_len == (self.db_len - 1)): ### TODO check longueur db
					# Every episode update the target model to be same with model
					self.agent.update_target_model()
					# Save model for each episode
					if self.agent.train:
						self.agent.save_model(f"{config.main_folder}/model_cache/{self.args.model}") ### TODO: faire un truc propre avec os
					if self.args.save:
						save_memory_db(self.episode_memory, self.general_infos, e)
					print(f"episode: {e} memory length: {len(self.agent.memory)} epsilon: {self.agent.epsilon} episode length: {episode_len}")
				
				# Updating state variables
				state = new_state
				preprocessed_state = new_preprocessed_state

			if self.agent.train:
				self.agent.train_on_memory(self.memory)

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
	parser.add_argument('--agent', type=str, default="DDQN",
						help='Choice of reinforcement Learning Agent', choices=["DDQN", "SAC"])
	parser.add_argument('--no_sim', type=str, default=False,
						help='agent uses stored database to train')
	parser.add_argument('--save', action="store_true",
						help='Saving each episode in a pickle file')
	parser.add_argument('--supervised', action="store_true",
						help='Use Human Player instead of Neural Player')
	args = parser.parse_args()
	if args.supervised:
		human_player = HumanPlayer(args)
	else:
		neural_player = NeuralPlayer(args)
