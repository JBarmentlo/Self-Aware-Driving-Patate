from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
# from tensorflow.keras.initializers import normal, identity
from tensorflow.keras.initializers import identity
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
# from tensorflow.keras import backend as K
from tensorflow.compat.v1.keras import backend as K
from collections import deque
import numpy as np
import random

#TODO: use output_shape from arg (HyperParams Opti)
turn_bins = 7

## Utils Functions ##
def linear_bin(a):
	#TODO: remove function
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
	#TODO: remove function
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


class DQNAgent:
	def __init__(self, state_size, action_space, input_shape, output_size, train=True):
		self.t = 0
		self.max_Q = 0
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
			self.epsilon = 0.0
			self.initial_epsilon = 0.0
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
		# Copy the model to target model
		# --> initialize the target model so that the parameters of model & target model to be same
		self.update_target_model()

	def build_model(self):
		model = Sequential()
		model.add(Conv2D(24, (5, 5), strides=(2, 2), padding="same",
                   input_shape=self.input_shape))  # 80*80*4
		model.add(Activation('relu'))
		model.add(Conv2D(32, (5, 5), strides=(2, 2), padding="same"))
		model.add(Activation('relu'))
		model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same"))
		model.add(Activation('relu'))
		model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
		model.add(Activation('relu'))
		model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same"))
		model.add(Activation('relu'))
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation('relu'))
		# 15 categorical bins for Steering angles
		model.add(Dense(self.output_size, activation="linear"))
		adam = Adam(lr=self.learning_rate)
		model.compile(loss='mse', optimizer=adam)
		return model

	def update_target_model(self):
		self.target_model.set_weights(self.model.get_weights())
	# Get action from model using epsilon-greedy policy

	def get_action(self, s_t):
		if np.random.rand() <= self.epsilon:
			return self.action_space.sample()[0]
		else:
			#print("Return Max Q Prediction")
			q_value = self.model.predict(s_t)
			# Convert q array to steering value
			return linear_unbin(q_value[0])

	def replay_memory(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def update_epsilon(self):
		if self.epsilon > self.epsilon_min:
			self.epsilon -= (self.initial_epsilon - self.epsilon_min) / self.explore

	def train_replay(self):
		if len(self.memory) < self.train_start:
			return
		batch_size = min(self.batch_size, len(self.memory))
		minibatch = random.sample(self.memory, batch_size)
		state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
		state_t = np.concatenate(state_t)
		state_t1 = np.concatenate(state_t1)
		targets = self.model.predict(state_t)
		self.max_Q = np.max(targets[0])
		target_val = self.model.predict(state_t1)
		target_val_ = self.target_model.predict(state_t1)
		for i in range(batch_size):
			if terminal[i]:
				targets[i][action_t[i]] = reward_t[i]
			else:
				a = np.argmax(target_val[i])
				targets[i][action_t[i]] = reward_t[i] + \
					self.discount_factor * (target_val_[i][a])
		self.model.train_on_batch(state_t, targets)

	def load_model(self, name):
		self.model.load_weights('model_cache/' + name)
	# Save the model which is under training

	def save_model(self, name):
		self.model.save_weights('model_cache/' + name)
