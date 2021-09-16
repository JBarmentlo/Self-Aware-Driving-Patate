import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import torch
from SAC import SoftActorCritic
# from SAC.Desesperate import SACAgent
from torch.utils.data import Dataset
from matplotlib.collections import LineCollection
from config import config

def distance(a, b):
	xs = a[0], b[0]
	ys = a[1], b[1]
	distance = ((xs[1] - xs[0]) ** 2 + (ys[1] - ys[0]) ** 2) ** (1/2)
	# distance = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** (1/2)
	return distance

class SimpleGame():
	def __init__(self):
		self.objective = (0., 0.)
		self.position = (0., 0.)
		self.history = []
		self.history_reward = []
		self.size = 10
		self.max_len = self.size * 5

		self.precision = 1
		self.t = 0

		# x,y
		self.map_max = (self.size, self.size)
		# x,y
		self.map_min = (0, -self.size)
		self.plot_init()

	def reset(self):
		x = np.random.random() * self.size
		y = np.random.random() * self.size * np.random.choice([-1., 1.])
		self.objective = (x, y)
		self.history = []
		self.history_reward = []
		self.t = 0
		my_x = ((self.map_max[0] - self.map_min[0]) / 2) + self.map_min[0]
		my_y = ((self.map_max[1] - self.map_min[1]) / 2) + self.map_min[1]
		self.position = (my_x, my_y)

		return self.step((0., 0.))

	def sample_actions(self):
		x = np.random.random()
		y = np.random.random() * np.random.choice([-1., 1.])
		return x, y

	
	def best_actions(self):
		x = self.objective[0] - self.position[0]
		y = self.objective[1] - self.position[1]
	
		x = max(-1., x)
		x = min(1., x)

		y = max(-1., y)
		y = min(1., y)

		return x, y

	def step(self, action_t):
		x_, y_ = action_t
		x, y = self.position
		self.position = (x + x_, y + y_)
		self.history.append(self.position)
		state_t1 = self._get_state()

		reward = self._reward((x_, y_))

		self.history_reward.append(reward)
		done = self._done()
		self.t += 1
		return state_t1, reward, done, None

	def _get_state(self):
		if len(self.history) < 2:
			second = self.history[-1]
		else:
			second = self.history[-2]
		d = distance(self.objective, self.position)
		state = d, self.t, *self.position, *second, *self.objective
		return state

	def _reward(self, action):
		if len(self.history) < 2:
			return 0.
		if self._is_win():
			return 100.
		if self._is_loose():
			return -100.
		a = distance(self.history[-1], self.objective)
		b = distance(self.history[-2], self.objective)
		# print(f"Dists: {a} < {b}")
		if a < b:
			return 1. + float(max(action))
		else:
			return -1.
		r = b - a
		return r + self._is_win()

	def _is_win(self):
		x = math.isclose(self.position[0], self.objective[0], abs_tol=self.precision)
		y = math.isclose(self.position[1], self.objective[1], abs_tol=self.precision)
		if x and y:
			# print("WIN")
			return 1
		return 0

	def _is_loose(self):
		if self.t >= self.max_len:
			return 1
		if self.position[0] < self.map_min[0]:
			return 1 
		if self.position[1] < self.map_min[1]:
			return 1
		if self.position[0] > self.map_max[0]:
			return 1
		if self.position[1] > self.map_max[1]:
			return 1
		return 0

	def _done(self):
		if self._is_win():
			return 1
		if self._is_loose():
			return -1
		return 0

	def plot(self):
		self.ax.clear()
		self.ax.set_xlim((self.map_min[0] - 1, self.map_max[0] + 1))
		self.ax.set_ylim((self.map_min[1] - 1, self.map_max[1] + 1))
		xs = [pos[0] for pos in self.history]
		ys = [pos[1] for pos in self.history]


		points = np.array([xs, ys]).T.reshape(-1, 1, 2)
		segments = np.concatenate([points[:-1], points[1:]], axis=1)


		# dydx = np.array(self.history_reward)


		# n_dydx = (dydx - dydx.min()) / dydx.max()
		# my_cm = cm.get_cmap("bwr")
		colors = ["g" if reward >= 1. else "r" for reward in self.history_reward]

		lc = LineCollection(segments, colors=colors[1:])


		line = self.ax.add_collection(lc)

		# self.ax.plot(xs, ys, c=)
		self.ax.scatter(*self.objective, color="red")
		plt.pause(1e-3)
		plt.show()

	def plot_exit(self):
		plt.ioff()
		plt.close()
	
	def plot_init(self):
		plt.ion()
		# plt.figure(figsize=(20, 8))
		fig, ax = plt.subplots()
		self.ax = ax

def stack(state: torch.Tensor, history = []) -> torch.Tensor:
	stack_size = 4
	state = torch.tensor(state).type(torch.float32)
	return state, history
	# print(f"{state.shape = }")
	if history == []:
		for _ in range(stack_size):
			history.append(state.to(torch.float32))
	else:
		history = history[1:stack_size]
		history.append(state.to(torch.float32))
	
	stacked_state = torch.stack(tuple(history), axis = 0)
	# stacked_state = torch.moveaxis(stacked_state, 1, 0)
	# print(f"{stacked_state.shape = }")
	return stacked_state, history


class SACDataset(Dataset):
	def __init__(self):
		self.data = []

	def add(self, processed_state, action, new_processed_state, reward, done):
		data = [processed_state, action, new_processed_state, reward, done]
		self.data.append(data)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, i):
		return self.data[i]


def loop_mine():
	game = SimpleGame()
	sac = SoftActorCritic(config)
	memory = SACDataset()

	buffer_maxlen = 1000
	visu_update = 200
	cool = []

	for i in range(100_000_000):
		hist = []
		state, reward, done, _ = game.reset()
		st, hist = stack(state, hist)

		while not done:
			a_t = sac.get_action(torch.unsqueeze(st, 0))
			state_t1, reward, done, _ = game.step(a_t[0])
			st_1, hist = stack(state_t1, hist)
			memory.add(st, a_t[0].detach().to(torch.float32), st_1, reward, done)
			st = st_1
			if i % visu_update == 0:
				game.plot()
		cool.append(np.mean(game.history_reward))

		if len(memory) >= buffer_maxlen // 100 and i % visu_update == 0:
			if i > 5:
				idx = min(visu_update, i - 1)
				print(f"Mean rewards for lasts {idx} = {np.mean(cool[-idx:])} @ {i}")
			if sac.train(memory):
				memory = SACDataset()
	game.plot_exit()


# def loop_his():
# 	# Last -***** at 2141*1000 episode
# 	# Last 0.**** at 2151*1000 episode
# 	# First 10.** at 2812*1000 episode
# 	# Best is 18.36 @ 4453
# 	game = SimpleGame()
# 	buffer_maxlen = 10000
# 	sac = SACAgent(None,
# 					config.discount_factor,
# 					config.Qfunction.tau,
# 					config.policy.alpha,
# 					config.Qfunction.lr,
# 					config.policy.lr,
# 					config.lr,
# 				buffer_maxlen)

# 	visu_update = 1000
# 	cool = []

# 	for i in range(100_000_000):
# 		hist = []
# 		state, reward, done, _ = game.reset()
# 		while not done:
# 			# state = torch.tensor(state)
# 			# print(state)
# 			a_t = sac.get_action(state)
# 			# print(a_t)
# 			state_t1, reward, done, _ = game.step(a_t)
# 			# state_t1 = torch.tensor(state_t1)
# 			sac.replay_buffer.push(state,
# 					a_t,
# 					reward, 
# 					state_t1,
# 					done)
# 			state = state_t1
# 			if i % visu_update == 0:
# 				game.plot()
# 		cool.append(np.mean(game.history_reward))
# 		if len(sac.replay_buffer) >= buffer_maxlen // 100 and i % visu_update == 0:
# 			if i > 5:
# 				idx = min(visu_update, i - 1)
# 				print(f"Mean rewards for lasts {idx} = {np.mean(cool[-idx:])} @ {i}")
# 			sac.update(config.batch_size)
# 	game.plot_exit()


def loop_best():
	# Best converge around 20.4
	game = SimpleGame()
	visu_update = 1000
	cool = []
	for i in range(100_000_000):
		_, _, done, _ = game.reset()

		while not done:
			_, _, done, _ = game.step(game.best_actions())
			if i % visu_update == 0:
				game.plot()
		cool.append(np.mean(game.history_reward))
		if i > 5:
			idx = min(visu_update, i - 1)
			print(f"Mean rewards for lasts {idx} = {np.mean(cool[-idx:])} @ {i}")
	game.plot_exit()


def loop_rand():
	# Random converge around -6.2
	game = SimpleGame()
	visu_update = 1000
	cool = []

	for i in range(100_000_000):
		_, _, done, _ = game.reset()
		while not done:
			_, _, done, _ = game.step(game.sample_actions())
			if i % visu_update == 0:
				game.plot()
		cool.append(np.mean(game.history_reward))
		if i > 5:
			idx = min(visu_update, i - 1)
			print(f"Mean rewards for lasts {idx} = {np.mean(cool[-idx:])} @ {i}")
	game.plot_exit()

if __name__ == "__main__":
	# loop_his()
	loop_best()
	loop_rand()
	loop_mine()
