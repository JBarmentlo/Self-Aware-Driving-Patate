import torch
from torch import nn
from torch import optim


class PolicyModel(nn.Module):
	def __init__(self):
		super(PolicyModel, self).__init__()
		self.lin_0 = nn.linear(128, 64)
		self.relu_0 = nn.ReLU(True)
		self.lin_1 = nn.linear(64, 32)
		self.relu_1 = nn.ReLU(True)
		self.lin_2 = nn.linear(32, 16)
		self.relu_2 = nn.ReLU(True)
		self.lin_3 = nn.linear(16, 8)
		self.relu_3 = nn.ReLU(True)
		self.lin_4 = nn.linear(8, 4)
		self.tanh_4 = nn.Tanh()

	def forward(self, x):
		x = self.lin_0(x)
		x = self.relu_0(x)
		x = self.lin_1(x)
		x = self.relu_1(x)
		x = self.lin_2(x)
		x = self.relu_2(x)
		x = self.lin_3(x)
		x = self.relu_3(x)
		x = self.lin_4(x)
		x = self.tanh_4(x)
		return x


class SACPolicy():
	def __init__(self, input_shape, bottleneck_shape, learning_rate=1e-3) -> None:
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model = PolicyModel().to(self.device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
		self.criterion = nn.MSELoss()
		self.loss = 0.

	def train(self, batch):
		batch = batch.to(self.device)
		self.optimizer.zero_grad()
		outputs = self.model(batch)
		loss = self.criterion(outputs, batch)
		loss.backward()
		self.optimizer.step()
		self.loss += loss.item()

	def predict(self, X):
		with torch.no_grad():
			X = X.to(self.device)
			Y = self.model(X)
			return Y

