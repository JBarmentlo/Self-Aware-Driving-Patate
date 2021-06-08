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
from copy import deepcopy
from agents.sac_policy import GaussianPolicy

def build_model_ValueNetwork(input_shape, output_size, learning_rate):
	"""
	This model will be an approximator of the Value Function to estimate the Expected Return of an episode from a state
	"""
	model = Sequential()
	model.add(Conv2D(24, (5, 5), strides=(2, 2), padding="same",
			   input_shape=input_shape))  # 80*80*4
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
	# TODO: check activation layer for output
	model.add(Dense(output_size, activation="linear"))
	adam = Adam(lr=learning_rate)
	# TODO: check which loss to choose
	model.compile(loss='mse', optimizer=adam)
	return model


class SoftActorCritic():
	"""
	Inspiration from: https://spinningup.openai.com/en/latest/algorithms/sac.html
	"""
	def __init__(self,
					state_size=1000,
					action_space=(2,),
					input_shape=(64, 64, 3),
					output_size=2,
					learning_rate=1e-4,
					train=True):
		print("Initialization of SAC")
		self.state_size = state_size
		self.action_space = action_space

		self.input_shape = input_shape
		self.output_size = output_size
		self.learning_rate = learning_rate
		self.train = train

		self.policy = GaussianPolicy()

		# Q functions estimators:
		self.phi_1 = build_model_ValueNetwork(input_shape, output_size, learning_rate)
		self.phi_2 = build_model_ValueNetwork(input_shape, output_size, learning_rate)

		self.discount_factor = 0.9
		self.lr_qfunc = 1e-4

	def policy_predict(self, s_t):
		pass

	def choose_action(self, s_t):
		pred = self.policy_predict(s_t)
		a_t = pred[random.randint(0, len(pred - 1))]
		return a_t

	def qfunc_predict(self, s_t1, a_t1, which=0):
			# Implementation is not clear if we need to sample a_t1 twice
			if which == 1:
				q_values = self.phi_1.predict(s_t1, a_t1)
			elif which == 2:
				q_values = self.phi_2.predict(s_t1, a_t1)
			else:
				q_values_1 = self.phi_1.predict(s_t1, a_t1)
				q_values_2 = self.phi_2.predict(s_t1, a_t1)
				q_values = np.min(q_values_1, q_values_2)
			return q_values

	def qfuncs_update(self, state_t, targets):
		# * The update should be on new networks to allow for soft update
		#	TODO: Update should be done with MSE
		phi_1 = deepcopy(self.phi_1)
		phi_2 = deepcopy(self.phi_2)

		phi_1.train_on_batch(state_t, targets)
		phi_2.train_on_batch(state_t, targets)

		return phi_i, phi_2


	def soft_net_update(self, net_old, net_new, TAU=0.8):
		# TODO: put TAU in config.py
		''' Update the targer gradually. '''
		# Extract parameters  
		net_old_params = net_old.named_parameters()
		net_new_params = net_new.named_parameters()
		
		dic_net_new_params = dict(net_new_params)

		for old_name, old_param in net_old_params:
			if old_name in old_param:
				dic_net_new_params[old_name].data.copy_(
					(TAU)*predi_param.data + (1-TAU)*old_param[predi_param].data)

		net_new.load_state_dict(net_old.state_dict())
		return net_new

	def compute_targets(self, r, s_t1, done):
		a_t1_throttle, a_t1_steering = self.policy.choose_action(s_t1)
		a_t1 = a_t1_steering
		# If done, eon (Expectation of ) is not necessary
		# 	If none might even break code
		eon = self.qfunc_predict(s_t1, a_t1, which=0) - \
						(self.lr_qfunc * np.ln(self.policy_predict(a_t1, s_t1)))
		targets = r + self.discount_factor  (1 - done) * eon
		return targets

	def train(self, replay_bufer, batch_size):
		for i in range(len(replay_bufer) // batch_size):
			# * Create batch
			batch = create_batch(replay_bufer, batch_size)
			state_t, action_t, reward_t, state_t1, done = batch

			# * Compute targets
			targets = self.compute_targets(reward_t, state_t1, done)

			# * Compute the update Q_functions estimators phi_1 & phi_2
			phi_1, phi_2 = self.qfuncs_update(state_t, targets)

			# * Update Policy, w/ gradient acent:
			# ? Not sure how to, reference back to pseudocode from here : https://spinningup.openai.com/en/latest/algorithms/sac.html#pseudocode
			# TODO: Once policy is implemented
			a_t = self.policy(s_t)

			# * Soft update the target networks
			self.soft_net_update(self.phi_1.model, phi_1.model)
			self.soft_net_update(self.phi_2.model, phi_2.model)


if __name__ == "__main__":
	SoftActorCritic()
