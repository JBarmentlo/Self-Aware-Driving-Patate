import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms


class ContractiveAutoEncoderModel(nn.Module):
	def __init__(self):
		super(ContractiveAutoEncoderModel, self).__init__()

		# self.fc1 = nn.Linear(784, 400, bias=False)  # Encoder
		self.fc1 = nn.Sequential(
					nn.Conv2d(3, 6, kernel_size=5),
					nn.ReLU(True),
					nn.Conv2d(6, 16, kernel_size=5),
					nn.ReLU(True))
		# self.fc2 = nn.Linear(400, 784, bias=False)  # Decoder
		self.fc2 = nn.Sequential(
					nn.ConvTranspose2d(16, 6, kernel_size=5),
					nn.ReLU(True),
					nn.ConvTranspose2d(6, 3, kernel_size=5),
					nn.ReLU(True))

		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		self.mse_loss = nn.BCELoss(size_average=False)

	def encoder(self, x):
		h1 = self.relu(self.fc1(x))
		return h1

	def decoder(self, z):
		h2 = self.sigmoid(self.fc2(z))
		return h2

	def forward(self, x):
			h1 = self.encoder(x)
			y = self.decoder(h1)
			return h1, y

	def criterion(self, W, x, recons_x, h, lam):
		"""Compute the Contractive AutoEncoder Loss
		Evalutes the CAE loss, which is composed as the summation of a Mean
		Squared Error and the weighted l2-norm of the Jacobian of the hidden
		units with respect to the inputs.
		See reference below for an in-depth discussion:
		#1: http://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder
		Args:
			`W` (FloatTensor): (N_hidden x N), where N_hidden and N are the
			dimensions of the hidden units and input respectively.
			`x` (Variable): the input to the network, with dims (N_batch x N)
			recons_x (Variable): the reconstruction of the input, with dims
			N_batch x N.
			`h` (Variable): the hidden units of the network, with dims
			batch_size x N_hidden
			`lam` (float): the weight given to the jacobian regulariser term
		Returns:
			Variable: the (scalar) CAE loss
		"""
		mse = self.mse_loss(recons_x, x)
		# Since: W is shape of N_hidden x N. So, we do not need to transpose it as
		# opposed to #1
		dh = h * (1 - h)  # Hadamard product produces size N_batch x N_hidden
		# Sum through the input dimension to improve efficiency, as suggested in #1

		# * These reshape are just to allow the code to run, they need to be carefully choosen 
		dh = dh.view(-1, 32)
		W = W.view(dh.shape[1], -1)


		w_sum = torch.sum(Variable(W)**2, dim=1)
		# unsqueeze to avoid issues with torch.mv
		w_sum = w_sum.unsqueeze(1)  # shape N_hidden x 1
		contractive_loss = torch.sum(torch.mm(dh**2, w_sum), 0)
		return mse + contractive_loss.mul_(lam)


class ContractiveAutoEncoder():
	def __init__(self, input_shape, bottleneck_shape, learning_rate=1e-3) -> None:
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model = ContractiveAutoEncoderModel().to(self.device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
		self.criterion = self.model.criterion
		self.loss = 0.
		self.lam = 1e-4

	def train(self, batch):
		batch = batch.to(self.device)
		self.optimizer.zero_grad()
		hidden, outputs = self.model(batch)
		# print(f"{batch.shape = }")
		# print(f"{self.model.state_dict().keys() = }")
		W = self.model.state_dict()['fc1.2.weight']
		# print(f"{outputs.shape = }")

		loss = self.criterion(W, batch, outputs, hidden, self.lam)

		loss.backward()
		self.optimizer.step()

		self.loss += loss.item()

	def predict(self, X):
		with torch.no_grad():
			X = X.to(self.device)
			_, Y = self.model(X)
			return Y

	def encode(self, X):
		with torch.no_grad():
			X = X.to(self.device)
			X_hat = self.model.encoder(X)
			return X_hat
