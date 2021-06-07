from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
# from tensorflow.keras.initializers import normal, identity
from tensorflow.keras.initializers import identity
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
# from tensorflow.keras import backend as K
from tensorflow.compat.v1.keras import backend as K
from collections import deque
import numpy as np
import random
from utils import linear_unbin, linear_bin


class DQNAgent:
	def __init__(self, state_size, action_space, input_shape, output_size, train=True):
		self.max_Q = 0.0
		self.train = train
		# Get size of state and action
		self.state_size = state_size
		self.action_space = action_space
		self.action_size = action_space
		self.input_shape = input_shape
		self.output_size = output_size
		# These are hyper parameters for the DQN
		self.discount_factor = 0.99
		self.learning_rate = 1e-4
		if (self.train):
			self.epsilon = 0.9
			self.initial_epsilon = 0.9
		else:
			self.epsilon = 1e-6
			self.initial_epsilon = 1e-6
		self.epsilon_min = 0.02
		self.batch_size = 64
		self.train_start = 100
		self.explore = 10000
		# Create replay memory using deque
		self.memory = deque(maxlen=10000)
		# Create main model and target model
		self.model = self.build_model()
		self.target_model = self.build_model()
		self.target_model.summary()
		# Copy the model to target model
		# --> initialize the target model so that the parameters of model & target model to be same
		self.update_target_model()

	def build_model(self):
		model = Sequential()
		model.add(Input(shape=(4, 128)))
		# (4, 128) -> comment les parser ?
		# On a besoin de se retrouver avec une dimension a la fin
		# - Layer de convolution le plus pertinent ?
		# - reshape pour mettre les vecteur a la suite
		model.add(Activation('relu'))
		model.add(Dense(128))
		model.add(Activation('relu'))
		model.add(Dense(128))
		model.add(Activation('relu'))
		model.add(Dense(128))
		model.add(Activation('relu'))
		model.add(Dense(128))
		model.add(Activation('relu'))
		model.add(Flatten())
		model.add(Dense(64))
		model.add(Activation('relu'))
		model.add(Dense(16))
		model.add(Activation('relu'))
		model.add(Dense(4))
		model.add(Activation('relu'))
		# 15 categorical bins for Steering angles
		model.add(Dense(self.output_size, activation="linear"))
		adam = Adam(lr=self.learning_rate)
		model.compile(loss='mse', optimizer=adam)
		return model

	def update_target_model(self):
		self.target_model.set_weights(self.model.get_weights())
	# Get action from model using epsilon-greedy policy

	def choose_action(self, s_t):
		if np.random.rand() <= self.epsilon:
			print("\tRandom choice !")
			return self.action_space.sample()[0]
		else:
			#print("Return Max Q Prediction")
			q_values = self.model.predict(s_t)
			# Convert q array to steering value
			print(f"\tModel 'True' prediction: {q_values.shape}")
			return linear_unbin(q_values[0])

	# def replay_memory(self, state, action, reward, next_state, done):
	# 	self.memory.append((state, action, reward, next_state, done))

	def update_epsilon(self):
		if self.epsilon > self.epsilon_min:
			self.epsilon -= (self.initial_epsilon - self.epsilon_min) / self.explore

	def load_model(self, name):
		self.model.load_weights(name)
	# Save the model which is under training

	def save_model(self, name):
		self.model.save_weights(name)

	def train_on_memory(self, memory):
		# print(f"Memory len: {len(memory)}")
		if len(memory) < self.train_start:
			return
		print(f"Train replay on {len(memory)} elements")
		# print(f"agent Batch size: {self.batch_size}")
		batch_size = min(self.batch_size, len(memory))
		# print(f"Batch size: {batch_size}")
		minibatch = random.sample(memory, batch_size)
		# For data structure look for comment in run_ddqn() 
		state_t, action_t, reward_t, state_t1, done, info = zip(*minibatch) ### TODO: add info
		state_t = np.concatenate(state_t) ### TODO: et dans le preprocessing?
		state_t1 = np.concatenate(state_t1)
		# Targets, are the predictions from agent.
		# Currently (april 20) they are 7 categories corresponding to values of steering
		# The agent predicts Q-Values for each of these categories
		targets = self.model.predict(state_t)
		# Use of agent.max_Q is for printing
		self.max_Q = np.max(targets)
		# We predict new state to be able to update Q-Values
		target_val_predict = self.model.predict(state_t1)
		target_val_update = self.target_model.predict(state_t1)
		for i in range(batch_size):
			# Here: we convert action_t which is a float directly used by simulator, to a category, which is what the model currently predicts
			# The last 0 is the angle, as for the moment we are not interested in the throttle
			# we use linear_bin, to convert float to categories
			bin_action = np.argmax(linear_bin(action_t[i][0]))
			if done[i]:
				targets[i][bin_action] = reward_t[i]
			else:
				# We get the most recent prediction from agent of the new_state obtained from action_t 
				a = np.argmax(target_val_predict[i])
				targets[i][bin_action] = reward_t[i] + \
					self.discount_factor * (target_val_update[i][a])
		# Now that all the targets have been updated, we can retrain the agent
		self.model.train_on_batch(state_t, targets)