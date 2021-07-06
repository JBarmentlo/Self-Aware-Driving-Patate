import os
import numpy as np
from collections import deque
from preprocessing import Preprocessing
from agents.ddqn import DQNAgent
from agents.sac import SoftActorCritic
from utils import is_cte_out, read_pickle_file, init_dic_info, append_db, save_memory_db
from Simulator import Simulator
from gym.spaces import Box
from config import config
import os
from s3 import S3
# doesn't show TF warnings..
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class NeuralPlayer():
	def __init__(self, args):
		'''
		run a DDQN training session, or test it's result, with the donkey simulator
		'''
		self.args = args
		self.our_s3 = None
		if self.args.save:
			if self.args.destination == "s3":
				self.our_s3 = S3()
			self.general_infos = init_dic_info(self.args, self.our_s3)
		if not self.args.no_sim:
			Simulator(self)
		# Construct gym environment. Starts the simulator if path is given.
		self.memory = deque(maxlen=10000)
		self.model_path = f"{config.main_folder}/model_cache/"
		self.model_name = self.args.model
		self.episode_memory = []
		self.db = None
		self.db_len = 0
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
								input_shape=(config.prep_img_rows, config.prep_img_cols, config.prep_img_channels),
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
			if self.args.sim == "simlaunch3000":
				self.client.kill_sim()
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
			# TODO: create function for following if/else
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
				if self.args.sim == "simlaunch3000":
					self.client.ping_sim()
				# Apply preprocessing and stack 4 frames
				preprocessed_state = self.prepare_state(state, old_state=None)
				# print(f"From env: cte {self.env.viewer.handler.cte}")
				# Choose action
				# TODO: It is time to make the model decide the throttle itself
				if not self.args.no_sim:
					if self.args.agent == "SAC":
						action = self.agent.choose_action(preprocessed_state)
						steering, _ = action
						# Adding throttle
						# ATTENTION: change was needed for SAC agent
						#		converted: 	[steering, throttle]
						#		to:			np.array([steering, throttle])
						action = np.array([steering, throttle])
						print(f"Steering: {steering:10.3} | Throttle: {throttle:10.3}")
					elif self.args.agent == "DDQN":
						steering = self.agent.choose_action(preprocessed_state)
						# Adding throttle
						action = [steering, throttle]
						# Do action
					new_state, reward, done, info = self.env.step(action)
					# episode_len > 10 because sometimes
					# 	simulator gives cte value from previous episode at the begining
					# TODO: create function for defining game_over
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
				
				# TODO: remove bc uncompatible with SAC
				self.agent.update_epsilon()
				# if self.agent.t % 30 == 0:
				# print(f"Episode: {e}, episode_len: {episode_len:<5} Action: [{action[0]:6.3} {action[1]:6.3}], Reward: {reward:6.4} Ep_len: {episode_len:<5} MaxQ: {self.agent.max_Q:3.3}")
				episode_len = episode_len + 1
				if done or (self.db_len != 0 and episode_len == (self.db_len - 1)): ### TODO check longueur db
					# Every episode update the target model to be same with model
					self.agent.update_target_model()
					# Save model for each episode
					if self.agent.train:
						self.agent.save_model(self.model_path, self.model_name)
						# self.agent.save_model(f"{config.main_folder}/model_cache/{self.args.model}") ### TODO: faire un truc propre avec os
					if self.args.save:
						save_memory_db(self.episode_memory, self.general_infos, e, self.our_s3)
					print(f"episode: {e} memory length: {len(self.memory)} epsilon: {self.agent.epsilon} episode length: {episode_len}")
				
				# Updating state variables
				state = new_state
				preprocessed_state = new_preprocessed_state

			if self.agent.train:
				self.agent.train_on_memory(self.memory)
