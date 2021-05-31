from tensorflow.keras import layers, activations, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
# from tensorflow.keras.initializers import normal, identity
from tensorflow.keras.initializers import identity
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
# from tensorflow.keras.layers import Conv2D, MaxPooling2D
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
	state_input = layers.Input(shape=input_shape[0])
	current_layer = layers.Conv2D(24, (5, 5), strides=(2, 2), padding="same", activation=activations.relu)(state_input)

	current_layer = layers.Activation('relu')(state_input)
	current_layer = layers.Conv2D(32, (5, 5), strides=(2, 2), padding="same", activation=activations.relu)(current_layer)
	current_layer = layers.Activation('relu')(current_layer)
	current_layer = layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same")(current_layer)
	current_layer = layers.Activation('relu')(current_layer)
	current_layer = layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same")(current_layer)
	current_layer = layers.Activation('relu')(current_layer)
	current_layer = layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same")(current_layer)
	current_layer = layers.Activation('relu')(current_layer)
	current_layer = layers.Flatten()(current_layer)
	current_layer = layers.Dense(512)(current_layer)
	state_end = layers.Activation('relu')(current_layer)

	action_input = layers.Input(shape=input_shape[1])
	# action_input_current_layer = layers.Dense(512)(action_input)
	# action_input_current_layer = layers.Activation('relu')(action_input_current_layer)

	merged = layers.Concatenate(axis=1)([state_end, action_input])

	current_layer = layers.Dense(50)(merged)
	current_layer = layers.Activation('relu')(current_layer)

	current_layer = layers.Dense(50)(current_layer)
	current_layer = layers.Activation('relu')(current_layer)

	current_layer = layers.Dense(50)(current_layer)

	# TODO: check activation layer for output

	output_size_throttle = output_size[0]
	output_size_steering = output_size[1]

	output_layer_throttle = layers.Dense(output_size_throttle, activation="sigmoid")(current_layer)
	output_layer_steering = layers.Dense(output_size_steering, activation="tanh")(current_layer)

	model = Model(inputs=[state_input, action_input], outputs=[output_layer_throttle, output_layer_steering])

	adam = Adam(lr=learning_rate)
	# TODO: check which loss to choose
	model.compile(loss='mse', optimizer=adam)

	return model
	# input1 = keras.layers.Input(shape=(1, ))
	# input2 = keras.layers.Input(shape=(1,))
	# merged = keras.layers.Concatenate(axis=1)([input1, input2])
	# dense1 = keras.layers.Dense(2, input_dim=2, activation=keras.activations.sigmoid, use_bias=True)(merged)
	# output = keras.layers.Dense(1, activation=keras.activations.relu, use_bias=True)(dense1)
	# model10 = keras.models.Model(inputs=[input1, input2], output=output)

	# model10.fit([array_1, array_2],output, batch_size=16, epochs=100)

class SoftActorCritic():
	"""
	Inspiration from: https://spinningup.openai.com/en/latest/algorithms/sac.html
	"""
	def __init__(self,
					state_size=1000,
					action_space=(2,),
					input_shape=(64, 64, 3),
					learning_rate=1e-4,
					train=True):
		print("Initialization of SAC")
		# Useless now, but needs to be compatible with DDQN
		self.state_size = state_size
		self.action_space = action_space
		self.epsilon = 0.99

		# Models characteristics
		self.train = train
		self.batch_size = 8

		# Policy
		self.input_shape_policy = input_shape
		self.learning_rate = learning_rate
		self.policy = GaussianPolicy(input_shape=input_shape, learning_rate=learning_rate)

		# Q functions estimators:
		self.lr_qfunc = 1e-4
		self.input_shape_phi_state = input_shape
		self.input_shape_phi_action = (2,)

		phi_input = (self.input_shape_phi_state, self.input_shape_phi_action)
		print(f"Input shape of ValueNet is: {phi_input}")
		# self.output_size_throttle = 1
		# self.output_size_steering = (1,)
		self.output_size = (1, 1)
		print(f"Output shape of 1tput_size {self.output_size}")

		self.phi_1 = build_model_ValueNetwork(phi_input, self.output_size, learning_rate)
		self.phi_2 = build_model_ValueNetwork(phi_input, self.output_size, learning_rate)
		self.phi_1.summary()
		self.discount_factor = 0.9

	def update_epsilon(self):
		pass

	def choose_action(self, s_t, concat=False):
		a_t_throttle, a_t_steering = self.policy.choose_action(s_t)
		a_t_throttle = np.squeeze(a_t_throttle)
		a_t_steering = np.squeeze(a_t_steering)
		# TODO: Make sure action are conscripted in 
		print(f"a_t_throttle: {a_t_throttle.shape} => {a_t_throttle}")
		print(f"a_t_steering: {a_t_steering.shape} => {a_t_steering}")
		# a_t_throttle = float(a_t_throttle)
		# a_t_steering = float(a_t_steering)
		if concat:
			a_t = np.concatenate([a_t_throttle.reshape(-1,1), a_t_steering.reshape(-1,1)], axis=1)
			return a_t
		return a_t_throttle, a_t_steering

	def qfunc_predict(self, s_t1, a_t1, which=0):
		# Implementation is not clear if we need to sample a_t1 twice
		print(f"Input shape state: {np.shape(s_t1)}")
		print(f"Input shape action: {np.shape(a_t1)}")
		if which == 1:
			q_values = self.phi_1([s_t1, a_t1])
		elif which == 2:
			q_values = self.phi_2([s_t1, a_t1])
		else:
			q_values_1 = self.phi_1([s_t1, a_t1])
			q_values_2 = self.phi_2([s_t1, a_t1])
			q_values = tf.math.minimum(q_values_1, q_values_2)
		return q_values

	def qfuncs_update(self, state_t, targets):
		# * The update should be on new networks to allow for soft update
		#	TODO: Update should be done with MSE
		phi_1 = deepcopy(self.phi_1)
		phi_2 = deepcopy(self.phi_2)

		print(f"qfuncs_update: Input shape: {state_t.shape},{targets.shape}")
		phi_1.train_on_batch([state_t, targets])
		phi_2.train_on_batch([state_t, targets])

		return phi_1, phi_2

	def update_target_model(self):
		# TODO: bad train_simulator interface
		# TODO:		The agent update itself at each train on memory
		pass
	
	def load_model(self, path, name):
		self.policy.actor_network.load_weights(path + "policy_" + name)
		self.policy.phi_1.load_weights(path + "phi_1_" + name)
		self.policy.phi_2.load_weights(path + "phi_2_" + name)
	# Save the model which is under training

	def save_model(self, path, name):
		self.policy.actor_network.save_weights(path + "policy_" + name)
		self.phi_1.save_weights(path + "phi_1_" + name)
		self.phi_2.save_weights(path + "phi_2_" + name)

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
		print(f"s_t1 shape: {np.shape(s_t1)}")
		a_t1 = self.choose_action(s_t1, concat=True)
		# actions = np.concatenate([action_throttle, action_steering], axis=1)
		print(f"a_t1 : {a_t1.shape}")
		# If done, eon (Expectation of ) is not necessary
		# 	If none might even break code
		# TODO Need to work through the shapes for getting the calcul right
		eon = self.qfunc_predict(s_t1, a_t1, which=0) - \
						(self.lr_qfunc * np.log(self.choose_action(s_t1, concat=True)))
		targets = r + self.discount_factor * (1 - done) * eon
		return targets

	def train_on_memory(self, replay_bufer):
		if len(replay_bufer) < self.batch_size:
			return
		for i in range(len(replay_bufer) // self.batch_size):
			# * Create batch
			batch = []
			for _ in range(self.batch_size):
				elem = replay_bufer.pop()
				# print(elem)
				batch.append(elem)
			
			state_t, action_t, reward_t, state_t1, done, _ = zip(*batch)

			state_t = np.concatenate(state_t, axis=0)
			action_t = np.concatenate(action_t, axis=0)
			# print(f"reward: {reward_t}")
			reward_t = np.array(reward_t)
			# print(f"reward: {reward_t}")
			state_t1 = np.concatenate(state_t1, axis=0)
			print(f"done: {done}")
			done = np.array(done).astype(int)
			print(f"done: {done}")

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
		replay_bufer.clear()


if __name__ == "__main__":
	SoftActorCritic()
