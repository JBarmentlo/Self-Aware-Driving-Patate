import torch
from torch import nn
from torch import optim

class AutoEncoderModel(nn.Module):
    def __init__(self):
        super(AutoEncoderModel, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(True))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 6, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6, 3, kernel_size=5),
            nn.ReLU(True))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class UndercompleteAutoEncoder():
	def __init__(self, input_shape, bottleneck_shape, learning_rate=1e-3) -> None:
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model = AutoEncoderModel().to(self.device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
		self.criterion = nn.MSELoss()
		self.loss = 0.

	def train(self, batch):
		batch = batch.to(self.device)
		self.optimizer.zero_grad()
		outputs = self.model(batch)
		# print(f"{batch.shape = }")
		# print(f"{outputs.shape = }")

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

