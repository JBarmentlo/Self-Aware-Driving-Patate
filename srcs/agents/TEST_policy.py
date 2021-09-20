# From https://github.com/woutervanheeswijk/example_continuous_control/blob/master/continuous_control
from SAC.Policy import Policy
from config import config

# Needed for training the network
import numpy as np

# Needed for animation
import matplotlib.pyplot as plt

import torch

VISU_UPDATE = 100
MAX_LEN = 1000

class TrackGauss():
	def __init__(self, ax, m_goal=0., s_goal=1.):
		self.ax = ax
		# mu
		self.a = []
		# reward
		self.r = []
		# mu goal
		self.m_g = m_goal
		# sigma goal
		self.s_g = s_goal
		self.max_reward = 1.

	def add_point(self, m, s):
		self.m.append(m)
		self.s.append(s)

	def reward(self, a):
		m_r = self.max_reward

		e = 1e-20
		reward = 1 / (abs(self.m_g - a)+e)

		# reward = m_r / max(self.s_g, abs(self.m_g-a)) * self.s_g
		self.a.append(a)
		self.r.append(reward)
		return reward

	def plot(self, epochs):
		self.ax.clear()

		max_len = MAX_LEN
		if len(epochs) > max_len:
			epochs = epochs[-max_len:]
			a = self.a[-max_len:]
		else:
			a = self.a

		# Plot outcomes
		size = len(epochs)
		self.ax.plot(epochs, [self.m_g] * size)#, label='mu', color=mu_color[0])
		upper = [self.m_g + self.s_g] * size
		dnner = [self.m_g - self.s_g] * size
		self.ax.fill_between(epochs, dnner, upper, alpha=.1)
		self.ax.plot(epochs, a, color="green")#, label='mu', color=mu_color[0])

		# self.ax.title.set_text('Throttle')



class PolicyTestor():
	def __init__(self):
		# Initialize fixed state
		s_t = np.ones(config.state_shape)
		self.state = torch.tensor(s_t).to(torch.float32)

		plt.ion()
		fig = plt.figure()
		self.ax = fig.subplots(1, 3)
		self.ax_1 = self.ax[0]
		self.ax_2 = self.ax[1]
		self.ax_3 = self.ax[2]

		self.throttle = TrackGauss(self.ax_1, m_goal=-50, s_goal=.1)
		# self.steering = TrackGauss(self.ax_2, m_goal=-0.5, s_goal=.1)

		self.policy = Policy(config.policy)
		self.r = []

	def Qvalues(self, state, action):
		a_0, a_1 = action[0]#.detach()
		# g_0, g_1 = gauss

		# m_0, s_0 = g_0
		# m_1, s_1 = g_1

		# self.throttle.add_point(m_0.detach(), s_0.detach())
		# self.steering.add_point(m_1.detach(), s_1.detach())

		reward_throttle = self.throttle.reward(a_0)
		# reward_steering = self.steering.reward(a_1)
		reward_steering = 0

		reward = reward_throttle + reward_steering

		self.r.append(reward)
		print(f"{reward = }")
		return reward


	def train_loop(self, nb_ite):
		epochs = []
		for i in range(nb_ite + 1):
			self.policy.train(torch.unsqueeze(self.state, 0), self.Qvalues)

			epochs.append(i)
			if i % 100 == 0:
				self.plot(epochs)

	def plot(self, epochs):

		self.throttle.plot(epochs)
		# self.steering.plot(epochs)


		max_len = MAX_LEN
		if len(epochs) > max_len:
			epochs = epochs[-max_len:]
			r_t = self.throttle.r[-max_len:]
			r = self.r[-max_len:]
		else:
			r_t = self.throttle.r
			r = self.r
		
		self.ax_3.clear()
		self.ax_3.plot(epochs, r, label='reward', color='green')
		self.ax_3.plot(epochs, r_t, label='reward', color='blue')
		# self.ax_3.plot(epochs, self.steering.r, label='reward', color='red')
		self.ax_3.title.set_text('Reward')

		plt.legend(loc='best')

		plt.pause(0.05)
		plt.show()

if __name__ == "__main__":
	test = PolicyTestor()
	test.train_loop(1_000_000)
