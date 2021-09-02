import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

class FlattenState(nn.Module):
	def __init__(self, input_channels):
		super(FlattenState, self).__init__()
		self.conv0 = nn.Conv1d(input_channels, 4, kernel_size=1, stride=1, padding=0)
		self.conv1 = nn.Conv1d(4, 2, kernel_size=1, stride=1, padding=0)
		self.conv2 = nn.Conv1d(2, 1, kernel_size=1, stride=1, padding=0)

		self.flatten = nn.Flatten()

	def forward(self, x):
		x = x.to(device)
		x = F.relu(self.conv0(x))
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = self.flatten(x)
		return x


class QfunctionModel(nn.Module):
	def __init__(self, input_shape):
		super(QfunctionModel, self).__init__()
		# Vector (4, 8)
		in_channels = input_shape[0]
		nb_features = input_shape[1] + 2

		self.FlatState = FlattenState(in_channels)
		
		self.dense1 = nn.Linear(nb_features, 16)
		self.dense2 = nn.Linear(16, 32)
		self.dense3 = nn.Linear(32, 32)
		self.dense4 = nn.Linear(32, 32)
		self.dense5 = nn.Linear(32, 1)


	def forward(self, state, action):
		state = state.to(device)
		action = action.to(device)

		state = self.FlatState(state)

		x = torch.cat((state, action), dim=1)

		x = F.relu(self.dense1(x))
		x = F.relu(self.dense2(x))
		x = F.relu(self.dense3(x))
		x = F.relu(self.dense4(x))
		x = self.dense5(x)

		return x


class Qfunction():
	def __init__(self, input_shape, bottleneck_shape, learning_rate=1e-3) -> None:
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model = QfunctionModel().to(self.device)
		self.target_model = QfunctionModel().to(self.device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
		self.criterion = nn.MSELoss()
		self.loss = 0.
		self.tau = 0.5

	def train(self, states, actions, targets):
		batch = batch.to(self.device)

		Qvalues = self.model(states, actions)

		loss = self.criterion(Qvalues, targets)

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