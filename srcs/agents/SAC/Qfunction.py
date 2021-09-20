import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from .Network import FlattenState, LinearDense

class QfunctionModel(nn.Module):
	def __init__(self, config):
		super(QfunctionModel, self).__init__()
		self.config = config
		if len(self.config.state_shape) == 2:
			in_channels = config.state_shape[0]
			nb_features = config.state_shape[1] + config.num_actions
			self.FlatState = FlattenState(in_channels)
			end_units = 128

		elif len(self.config.state_shape) == 1:
			nb_features = config.state_shape[0] + config.num_actions
			end_units = 128

		self.LinearDense = LinearDense(nb_features, end_units, 1)
		self.end = nn.Linear(end_units, config.num_actions)

		self.end.weight.data.uniform_(-3e-3, 3e-3)
		self.end.bias.data.uniform_(-3e-3, 3e-3)


	def forward(self, state, action):
		if len(self.config.state_shape) == 2:
			state = self.FlatState(state)
		# print(f"Forward: {state.shape = }")
		x = torch.cat((state, action), dim=1)
		x = self.LinearDense(x)
		x = self.end(x)
		return x


class Qfunction():
	def __init__(self, config) -> None:
		self.config = config
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = torch.device("cpu")
		# * Models creation
		# * Target model: is receiving training
		# * Local model: allows for prediction
		self.model = QfunctionModel(config).to(self.device)
		print(self.model)
		self.target_model = QfunctionModel(config).to(self.device)
		# * Training tools
		self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
		self.criterion = nn.MSELoss()
		# * Vars
		self.loss = 0.
		self.tau = config.tau

	def train(self, states, actions, targets):
		# print(f"Qfunc train")
		# print(f"{states.shape = }")
		# print(f"{actions.shape = }")
		states = states.to(self.device)
		actions = actions.to(self.device)
		targets = targets.to(self.device)

		Qvalues = self.target_model(states, actions)

		loss = self.criterion(Qvalues, targets)
		# print(f"{loss = }")

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		self.loss += loss.item()
	
	def soft_update(self):
		"""Soft update model parameters.
		θ_target = τ*θ_local + (1 - τ)*θ_target
		Params
		=======
			local model (PyTorch model): weights will be copied from
			target model (PyTorch model): weights will be copied to
			tau (float): interpolation parameter
		"""
		tau = self.tau

		for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
			target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)

		self.model.load_state_dict(self.target_model.state_dict())
