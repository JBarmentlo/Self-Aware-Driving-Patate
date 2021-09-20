import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import math
from torch.distributions import Normal
from .Network import FlattenState, LinearDense

class GaussianOut(nn.Module):
	def __init__(self, input_units, num_actions, log_std_min, log_std_max):
		super(GaussianOut, self).__init__()
		# mean
		self.mean_linear = LinearDense(input_units, num_actions, 1)

		self.mean_linear.layers[-1].weight.data.uniform_(-3e-3, 3e-3)
		self.mean_linear.layers[-1].bias.data.uniform_(-3e-3, 3e-3)
		# log std
		self.log_std_linear = LinearDense(input_units, num_actions, 1)
		self.log_std_linear.layers[-1].weight.data.uniform_(-3e-3, 3e-3)
		self.log_std_linear.layers[-1].bias.data.uniform_(-3e-3, 3e-3)
		# log std range
		self.log_std_min = log_std_min
		self.log_std_max = log_std_max
		
	def forward(self, x):
		mean    = self.mean_linear(x)
		log_std = self.log_std_linear(x)
		log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
		return mean, log_std


class PolicyModel(nn.Module):
	# * Great ressource: 
	# * https://towardsdatascience.com/in-depth-review-of-soft-actor-critic-91448aba63d4
	def __init__(self, config):
		super(PolicyModel, self).__init__()
		self.config = config
		self.build()

	def build(self):
		if len(self.config.state_shape) == 2:
			in_channels = self.config.state_shape[0]
			nb_features = self.config.state_shape[1]

			self.FlatState = FlattenState(in_channels)
			end_units = 4
		elif len(self.config.state_shape) == 1:
			nb_features = self.config.state_shape[0]
			end_units = 128

		self.LinearDense = LinearDense(nb_features, end_units, 1)
		self.gaussian_end = GaussianOut(end_units, 
										self.config.num_actions,
										self.config.log_std_max, 
										self.config.log_std_min)

	def forward(self, x):
		if len(self.config.state_shape) == 2:
			x = self.FlatState(x)
		x = F.relu(self.LinearDense(x))
		mean, log_std = self.gaussian_end(x)
		return mean, log_std


	def sample(self, state, epsilon=1e-6):
		# print(f"PLOP: {state.shape = }")
		mean, log_std = self.forward(state)

		std = log_std.exp()
		# print(f"{std = }")
		# print(f"{mean = }")

		normal = Normal(mean, std)
		z = normal.rsample()
		# print(f"{z = }")
		action = torch.tanh(z)

		log_pi = normal.log_prob(z) - torch.log(1 - action.pow(2) + self.config.epsilon)
		# log_pi = log_pi.sum(1, keepdim=True)

		return mean, std, action, log_pi


class Policy():
	def __init__(self, config) -> None:
		self.config = config
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = torch.device("cpu")
		self.model = PolicyModel(config).to(self.device)
		print(self.model)
		self.learning_rate = config.lr
		self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
		self.loss = 0.

		# entropy temperature
		self.alpha = config.alpha
		self.alpha_lr = config.alpha

		self.target_entropy = - \
			torch.prod(torch.Tensor(self.config.num_actions).to(self.device)).item()
		self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
		self.alpha_optim = optim.Adam([self.log_alpha], lr=self.config.alpha_lr)


	def train(self, state_t, phi_min):
		state_t = state_t.to(self.device)

		loss = self.criterion(state_t, phi_min)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		# self.loss += loss.item()

	def criterion(self, state, phi_min):
		_, _, action, log_pi = self.model.sample(state)

		alpha = self.alpha
		Qvalue = phi_min(state, action)
		loss = (alpha * log_pi - Qvalue)
		
		return loss.mean()

	def update_temperature(self, state):
		_, _, _, log_pi = self.model.sample(state)
		alpha_loss = (self.log_alpha * (-log_pi -
		              self.target_entropy).detach()).mean()

		self.alpha_optim.zero_grad()
		alpha_loss.backward()
		self.alpha_optim.step()

		self.alpha = self.log_alpha.exp()
