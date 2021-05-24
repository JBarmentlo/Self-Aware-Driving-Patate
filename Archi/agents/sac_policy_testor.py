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
		self.mu_ar = [[], []]
		self.sigma_ar = [[], []]
		self.reward_ar = []
		self.target_ar = [[], []]
		self.range_ar = [[], []]

		# Define properties reward function
		self.mu_target_throttle = 4.0
		self.target_range_throttle = 0.25
		self.mu_target_steering = 10.0
		self.target_range_steering = 0.75
		self.max_reward = 1.0

		# bias 0.0 yields mu=0.0 with linear activation function
		# bias 0.55 yields sigma=1.0 with softplus activation function
		self.Pol = GaussianPolicy(input_shape=(1,),
									learning_rate=0.001)

		# Printing
		self.ax = None
		self.ax_1 = None

	def compute_reward(self, action, mu_target, target_range):
		reward = self.max_reward / \
						max(target_range, \
							abs(mu_target-action)) \
					* target_range
		return reward


	def train_loop(self, nb_ite):
		for i in range(nb_ite + 1):
			# Draw action from policy
			actions = self.Pol.choose_action(self.state)

			# * Compute reward
			reward_throttle = self.compute_reward(actions[0],
												self.mu_target_throttle,
												self.target_range_throttle)
			reward_steering = self.compute_reward(actions[1],
												self.mu_target_steering,
												self.target_range_steering)
			
			reward = reward_throttle + reward_steering

			self.Pol.update(self.state, actions, reward)
			

			# Update console output and plot
			if np.mod(i, 100) == 0:
				print('\n====== episode', i, '======')
				print(f"mu:      {float(self.Pol.mu_throttle):.5} | {float(self.Pol.mu_steering):.5}")
				print(f"sigma:   {float(self.Pol.sigma_throttle):.5} | {float(self.Pol.sigma_steering):.5}")
				print(f"actions: {float(actions[0]):.5} | {float(actions[1]):.5}")
				print(f'reward:  {float(reward)}')
				print(f'loss:    {float(self.Pol.loss_)}')
				print()

				# Append arrays
				self.epoch_ar.append(int(i))
				self.mu_ar[0].append(float(self.Pol.mu_throttle))
				self.mu_ar[1].append(float(self.Pol.mu_steering))

				self.sigma_ar[0].append(float(self.Pol.sigma_throttle))
				self.sigma_ar[1].append(float(self.Pol.sigma_steering))

				self.target_ar[0].append(float(self.mu_target_throttle))
				self.target_ar[1].append(float(self.mu_target_steering))

				self.range_ar[0].append(float(self.target_range_throttle))
				self.range_ar[1].append(float(self.target_range_steering))

				self.reward_ar.append(float(reward))

				self.plot()

	def plot(self):
		if self.ax_1 == None:
			plt.ion()
			fig = plt.figure()
			self.ax = fig.subplots(1, 3)
			# print(self.ax)
			self.ax_1 = self.ax[0]
			self.ax_2 = self.ax[1]
			self.ax_3 = self.ax[2]
			# self.ax_4 = self.ax[1][1]
		else:
			self.ax_1.clear()
			self.ax_2.clear()
			self.ax_3.clear()

		mu_color = ['#cd9b9b', '#ff9b9b']
		si_color = ['#9b9bcd', '#9b9bff']

		# Plot outcomes
		self.ax_1.plot(self.epoch_ar, self.mu_ar[0], label='mu', color=mu_color[0])
		self.ax_1.plot(self.epoch_ar, self.target_ar[0], label='mu_target', color=mu_color[1])
		self.ax_1.plot(self.epoch_ar, self.sigma_ar[0], label='sigma', color=si_color[0])
		self.ax_1.plot(self.epoch_ar, self.range_ar[0], label='range_target', color=si_color[1])
		self.ax_1.title.set_text('Throttle')

		self.ax_2.plot(self.epoch_ar, self.mu_ar[1], label='mu', color=mu_color[0])
		self.ax_2.plot(self.epoch_ar, self.target_ar[1], label='mu_target', color=mu_color[1])
		self.ax_2.plot(self.epoch_ar, self.sigma_ar[1], label='sigma', color=si_color[0])
		self.ax_2.plot(self.epoch_ar, self.range_ar[1], label='range_target', color=si_color[1])
		self.ax_2.title.set_text('Steering')

		self.ax_3.plot(self.epoch_ar, self.reward_ar, label='reward', color='green')
		self.ax_3.title.set_text('Reward')

		# Add labels and legend
		# plt.xlabel('Episode')
		# plt.ylabel('Parameter value')
		# plt.grid()
		# plt.legend(loc='best')

		plt.pause(0.05)
		plt.show()

if __name__ == "__main__":
	test = SacTestor()
	test.train_loop(1_000_000)
