import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers, activations, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
# from tensorflow.keras.initializers import normal, identity
from tensorflow.keras.initializers import identity
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
# from tensorflow.keras.layers import Conv2D, MaxPooling2D
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
	current_layer = layers.Conv2D(24, (5, 5), 
								strides=(2, 2), padding="same", 
								activation=activations.relu)(state_input)
	current_layer = layers.Conv2D(32, (5, 5), 
								strides=(2, 2), padding="same", 
								activation=activations.relu)(current_layer)
	current_layer = layers.Conv2D(64, (5, 5), 
								strides=(2, 2), padding="same", 
								activation=activations.relu)(current_layer)
	current_layer = layers.Conv2D(64, (3, 3), 
								strides=(2, 2), padding="same",
								activation=activations.relu)(current_layer)
	current_layer = layers.Conv2D(64, (3, 3), 
								strides=(1, 1), padding="same", 
								activation=activations.relu)(current_layer)
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

	# TODO: check activation layer for output -> 
	# 	Bad Today -> It's Q-Values, not actions !!
	# 	Q Values: 
	#		- Do they have same characteristics for both ?
	# 			Yes
	# 		- Can it go to infinity ?
	# 			Yes
	# 		- Can it be neg ?
	# 			Unsure

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
		self.lr_qfunc = 1e-3
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
			# * It's in the shape (nb_examples, nb_targets)
			q_values = tf.reshape(q_values, (-1, 2))
		return q_values

	def qfuncs_update(self, state_t, action_t, targets):
		# * The update should be on new networks to allow for soft update
		#	TODO: Update should be done with MSE

		# TODO: Okay so what I will do is HORRIBLE and TEMPORARY
		# My problem is I need to compute a soft update between the old and new model
		# 	Right now, i don't know how to do so in a efficient manner
		#		As I want this exemple running i will do it in a REALLY
		#		inefficent manner, but I WILL make it better later
		phi_1 = tf.keras.models.clone_model(self.phi_1)
		phi_2 = tf.keras.models.clone_model(self.phi_2)
		adam = Adam(lr=self.lr_qfunc)
		phi_1.compile(loss='mse', optimizer=adam)
		adam = Adam(lr=self.lr_qfunc)
		phi_2.compile(loss='mse', optimizer=adam)

		print(
			f"qfuncs_update: Input shape: ({state_t.shape},{action_t.shape})")
		q_val_throttle = targets[:,0]
		q_val_steering = targets[:,1]
		print(f"qfuncs_update: output shape: ({q_val_throttle.shape},{q_val_steering.shape})")
		phi_1.train_on_batch([state_t, action_t], [
								q_val_throttle, q_val_steering])
		phi_2.train_on_batch([state_t, action_t], [q_val_throttle, q_val_steering])

		return phi_1, phi_2

	def update_target_model(self):
		# TODO: bad train_simulator interface
		# *			Because -> The agent update itself at each train on memory
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
		print(type(net_old))
		print(type(net_new))

		# Resources:
		# 	https://github.com/tensorflow/tensorflow/issues/23906
		# 	https://stackoverflow.com/questions/51354186/how-to-update-weights-manually-with-keras

		with tf.Session() as sess:
			def hello(scopeName='sc1'):
				with tf.variable_scope(scopeName):
					a1 = tf.get_variable(name='test_var1', initializer=0.)
					b1 = tf.get_variable(name='test_var2', initializer=0.)

			g=hello('sc1')
			g2=hello('new')
			g3=hello('old')
			sess.run(tf.global_variables_initializer())

			t_vars=tf.trainable_variables()
			var_in_sc1= [var for var in t_vars if 'sc1' in var.name]
			var_in_new= [var for var in t_vars if 'new' in var.name]
			var_in_old= [var for var in t_vars if 'old' in var.name]
			
			print('save 0')
			for var_idx, var in enumerate(var_in_sc1):
				var_in_old[var_idx].load(var_in_sc1[var_idx].eval(),sess)
			print('add 1')
			for var in var_in_sc1:
				sess.run(tf.assign_add(var,1.))

			print('save 1')
			for var_idx, var in enumerate(var_in_sc1):
				var_in_new[var_idx].load(var_in_sc1[var_idx].eval(),sess)
			print('add 1')
			for var in var_in_sc1:
				sess.run(tf.assign_add(var,1.))
			print(sess.run(var_in_sc1))

			print('load old')
			for var_idx, var in enumerate(var_in_old):
				var_in_sc1[var_idx].load(var_in_old[var_idx].eval(),sess)
			print(sess.run(var_in_sc1))

			print('load new')
			for var_idx, var in enumerate(var_in_new):
				var_in_sc1[var_idx].load(var_in_new[var_idx].eval(),sess)
			print(sess.run(var_in_sc1))
	
		print(net_old.trainable_variables)
		print(net_new.trainable_variables)

		for i, otv in enumerate(net_old.trainable_variables):
			net_new.trainable_variables[i] = ((TAU)*net_new.trainable_variables[i] + (1-TAU)*otv)


		# # Get all the variables in the Q primary network.
		# q_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Q_primary")
		# # Get all the variables in the Q target network.
		# q_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Q_target")
		# assert len(q_vars) == len(q_target_vars)

		# def update_target_q_net_soft(tau=0.05):
		# 	# Soft update: polyak averaging.
		# 	sess.run([v_t.assign(v_t * (1. - tau) + v * tau) for v_t, v in zip(q_target_vars, q_vars)])

		# net_old_params = net_old.named_parameters()
		# net_new_params = net_new.named_parameters()
		
		# dic_net_new_params = dict(net_new_params)

		# for old_name, old_param in net_old_params:
		# 	if old_name in old_param:
		# 		dic_net_new_params[old_name].data.copy_(
		# 			(TAU)*predi_param.data + (1-TAU)*old_param[predi_param].data)

		# net_new.load_state_dict(net_old.state_dict())
		return net_new

	def compute_targets(self, r, s_t1, done):
		print(f"s_t1 shape: {np.shape(s_t1)}")
		a_t1 = self.choose_action(s_t1, concat=True)
		# actions = np.concatenate([action_throttle, action_steering], axis=1)
		print(f"a_t1 shape : {a_t1.shape}")
		print(f"a_t1 : {a_t1}")

		# When done, eon (Expectation of n? (next action maybe)) is not necessary
		# 	Be carefull to check if it can be none and if it can break code

		# FROM formula line 12 here: https://spinningup.openai.com/en/latest/algorithms/sac.html#pseudocode
		# OH MY GOODNESS the LOG IS ABOUT THE PROBABILITY OF DRAWING THE ACTIONS,
		# 	NOT THE ACTIONS ITSELF !!!!!!!!!!!
		# Will need to adapt, for the moment I consider every action has a probability of one,
		# 	Once this agent can run without generating Exceptions
		# 	I will apply the changes correctly in SAC_policy 
		probability_to_draw_action = self.policy.policy_probability(s_t1)
		lr_action = (self.lr_qfunc * np.log(probability_to_draw_action))
		
		print(f"lr_action shape: {lr_action.shape}")
		print(f"lr_action : {lr_action}")
		pred_q = self.qfunc_predict(s_t1, a_t1, which=0)
		print(f"pred_q shape: {pred_q.shape}")
		print(f"pred_q : {pred_q}")
		eon = pred_q - lr_action
		eon = tf.cast(eon, tf.float64)
		print(f"eon : {eon}")
		print(f"done shape: {done.shape}")

		print(f"done: {done}")
		done = tf.constant(done, dtype=tf.float64)
		print(f"done: {done}")
		print(f"self.discount_factor: {self.discount_factor}")
		discount_factor = tf.constant(self.discount_factor, dtype=tf.float64)
		print(f"discount_factor: {discount_factor}")
		on_off_grad = discount_factor * (1 - done)
		print(f"on_off_grad: {on_off_grad}")
		on_off_grad = tf.reshape(on_off_grad, (-1, 1))
		print(f"on_off_grad reshape: {on_off_grad}")
		on_off_grad = tf.concat([on_off_grad, on_off_grad], axis=1)
		on_off_grad = tf.cast(on_off_grad, tf.float64)
		print(f"on_off_grad concat: {on_off_grad}")
		# on_off_grad = on_off_grad.numpy().tolist()
		# print(on_off_grad)
		# on_off_grad = tf.constant(on_off_grad)
		# print(on_off_grad)
		# on_off_grad = tf.cast()
		print(f"on_off_grad shape: {on_off_grad.shape}")
		print(f"eon shape: {eon.shape}")
		print(eon)
		r = tf.constant(r)
		r = tf.reshape(r, (-1, 1))
		r = tf.concat([r, r], axis=1)
		print(f"r shape: {r.shape}")
		print(r)
		targets = tf.add(r, tf.multiply(on_off_grad, eon))
		print(f"targets shape: {targets.shape}")
		print(targets)
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
			action_t = np.array(action_t)
			print(f"Action_t: {action_t}")
			print(f"Action_t shape: {action_t.shape}")
			# action_t = np.concatenate(action_t, axis=1)
			# print(f"reward: {reward_t}")
			reward_t = np.array(reward_t)
			# print(f"reward: {reward_t}")
			state_t1 = np.concatenate(state_t1, axis=0)
			print(f"done: {done}")
			done = np.array(done).astype(int)
			print(f"done: {done}")

			# line 12:
			# * Compute targets
			targets = self.compute_targets(reward_t, state_t1, done)

			# line 13:
			# * Compute the update Q_functions estimators phi_1 & phi_2
			phi_1, phi_2 = self.qfuncs_update(state_t, action_t, targets)

			# line 14:
			# * Update Policy, w/ gradient acent:
			# ? Not sure how to, reference back to pseudocode from here : https://spinningup.openai.com/en/latest/algorithms/sac.html#pseudocode
			# TODO: Once policy is implemented
			qvals = self.qfunc_predict(state_t, action_t)
			qvals = tf.cast(qvals, tf.float64)
			pol_prob = self.policy.policy_probability(state_t)
			pol_prob = tf.cast(pol_prob, tf.float64)

			policy_reward = qvals - pol_prob
			self.policy.update(state_t, action_t, policy_reward)

			# line 15:
			# * Soft update the target networks
			debug = False
			# debug = True
			if not debug:
				self.phi_1 = phi_1
				self.phi_2 = phi_2
			else:
				self.phi_1 = self.soft_net_update(self.phi_1, phi_1)
				self.phi_2 = self.soft_net_update(self.phi_2, phi_2)
		replay_bufer.clear()


if __name__ == "__main__":
	SoftActorCritic()
