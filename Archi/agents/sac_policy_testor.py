# From https://github.com/woutervanheeswijk/example_continuous_control/blob/master/continuous_control

from sac_policy import GaussianPolicy

# Needed for training the network
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.initializers as initializers

# Needed for animation
import matplotlib.pyplot as plt

class SacTestor():
	def __init__(self):
		# Initialize fixed state
		self.state = tf.constant([[1.0]])

		# Initialize arrays for plot
		self.epoch_ar = []
		self.mu_ar = []
		self.sigma_ar = []
		self.reward_ar = []
		self.target_ar = []

		# Define properties reward function
		self.mu_target = 4.0
		self.target_range = 0.25
		self.max_reward = 1.0

		# bias 0.0 yields mu=0.0 with linear activation function
		# bias 0.55 yields sigma=1.0 with softplus activation function
		self.Pol = GaussianPolicy(input_shape=(1,),
											bias_mu=0.0,
											bias_sigma=0.55,
											learning_rate=0.001)

		# Printing
		self.ax = None

	def train_loop(self, nb_ite):
		for i in range(nb_ite + 1):
			# Draw action from policy
			action = self.Pol.choose_action(self.state)

			# * Compute reward
			reward = self.max_reward / \
                            max(self.target_range, \
								abs(self.mu_target-action)) \
						* self.target_range

			self.Pol.update(self.state, action, reward)
			

			# Update console output and plot
			if np.mod(i, 100) == 0:

				print('\n======episode', i, '======')
				print('mu', float(self.Pol.mu_))
				print('sigma', float(self.Pol.sigma_))
				print('action', float(action))
				print('reward', float(reward))
				print('loss', float(self.Pol.loss_))

				# Append arrays
				self.epoch_ar.append(int(i))
				self.mu_ar.append(float(self.Pol.mu_))
				self.sigma_ar.append(float(self.Pol.sigma_))
				self.reward_ar.append(float(reward))
				self.target_ar.append(float(self.mu_target))

				self.plot()

	def plot(self):
		if self.ax == None:
			plt.ion()
			fig = plt.figure()
			self.ax = fig.add_subplot(1, 1, 1)
		else:
			self.ax.clear()

		# Plot outcomes
		self.ax.plot(self.epoch_ar, self.mu_ar, label='mu')
		self.ax.plot(self.epoch_ar, self.sigma_ar, label='sigma')
		self.ax.plot(self.epoch_ar, self.reward_ar, label='reward')
		self.ax.plot(self.epoch_ar, self.target_ar, label='target')

		# Add labels and legend
		plt.xlabel('Episode')
		plt.ylabel('Parameter value')
		plt.grid()
		plt.legend(loc='best')

		plt.pause(0.05)
		plt.show()

if __name__ == "__main__":
	test = SacTestor()
	test.train_loop(10_000)
