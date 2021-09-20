import torch
from torch import nn
from torch import optim

class AutoEncoderModel(nn.Module):
	def __init__(self):
		super(AutoEncoderModel, self).__init__()

		self.encoder = nn.Sequential(
			nn.Conv2d(3, 16, kernel_size=(3, 3)),
			nn.ReLU(True),
			nn.Conv2d(16, 32, kernel_size=(3, 3)),
			nn.ReLU(True),
			nn.Conv2d(32, 64, kernel_size=(3, 3)),
			nn.ReLU(True),
			nn.Conv2d(64, 128, kernel_size=(3, 3)),
			nn.ReLU(True))

		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(128, 64, kernel_size=(3, 3)),
			nn.ReLU(True),
			nn.ConvTranspose2d(64, 32, kernel_size=(3, 3)),
			nn.ReLU(True),
			nn.ConvTranspose2d(32, 16, kernel_size=(3, 3)),
			nn.ReLU(True),
			nn.ConvTranspose2d(16, 3, kernel_size=(3, 3)),
			nn.ReLU(True))

	def forward(self, x):
		x = self.encoder(x)
		print(f"{x.shape = }")
		x = self.decoder(x)
		return x


class UndercompleteAutoEncoder():
	def __init__(self, config) -> None:
		self.config = config
		self.model_path = f"{self.config.model_dir}{self.config.name}"
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model = AutoEncoderModel().to(self.device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
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

	def encode(self, X):
		with torch.no_grad():
			X = X.to(self.device)
			X = self.model.encoder(X)
			return X

	def load(self):
		self.model.load_state_dict(torch.load(self.model_path))
		self.model.eval()

	def save(self):
		torch.save(self.model.state_dict(), self.model_path)

