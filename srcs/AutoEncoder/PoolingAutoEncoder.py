from numpy.core.numeric import indices
import torch
from torch import nn
from torch import optim


class InMatrixLayer(nn.Module):
	def __init__(self, in_channels=3, out_channels=6, conv_kernel=5, pool_kernel=2):
		super(InMatrixLayer, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.conv_kernel = conv_kernel
		self.pool_kernel = pool_kernel
		self.build()

	def build(self):
		self.Conv2d = nn.Conv2d(self.in_channels,
								self.out_channels,
								kernel_size=self.conv_kernel)
		self.ReLU = nn.ReLU(True)
		self.MaxPool = nn.MaxPool2d(self.pool_kernel, return_indices=True)

	def forward(self, x):
		x = self.Conv2d(x)
		x = self.ReLU(x)
		shape = x.shape
		x, indices = self.MaxPool(x)
		return x, (shape, indices)

	def output_shape(self, input_shape: tuple) -> tuple:
		"""[summary]
		For a given shape, compute the outputted shape produced by the network

		Args:
			input_shape (tuple):
			input shape should be of len 3.  

		Returns:
			torch.Tensor
		"""

		pass


class AutoEncoderModel(nn.Module):
	def __init__(self):
		super(AutoEncoderModel, self).__init__()
		self._init_encoder()
		self._init_decoder()

	def _init_encoder(self):
		self.encoder_Conv2d_0 = nn.Conv2d(3, 6, kernel_size=5)
		self.encoder_ReLU_0 = nn.ReLU(True)
		self.encoder_MaxPool_0 = nn.MaxPool2d(2, return_indices=True)
		self.encoder_Conv2d_1 = nn.Conv2d(6, 16, kernel_size=5)
		self.encoder_ReLU_1 = nn.ReLU(True)
		self.encoder_MaxPool_1 = nn.MaxPool2d(2, return_indices=True)
		self.encoder_flatten_2 = nn.Flatten()
		self.encoder_dense_2 = nn.Linear(15984, 256)
		self.encoder_dense_3 = nn.Linear(256, 128)

	def _init_decoder(self):
		self.decoder_dense_m2 = nn.Linear(128, 256)
		self.decoder_dense_m1 = nn.Linear(256, 15984)
		self.decoder_MaxunPool_0 = nn.MaxUnpool2d(2)
		self.decoder_ConvT2d_0 = nn.ConvTranspose2d(16, 6, kernel_size=5)
		self.decoder_ReLU_0 = nn.ReLU(True)
		self.decoder_MaxunPool_1 = nn.MaxUnpool2d(2)
		self.decoder_ConvT2d_1 = nn.ConvTranspose2d(6, 3, kernel_size=5)
		self.decoder_ReLU_1 = nn.ReLU(True)

	def encoder(self, x):
		x = self.encoder_Conv2d_0(x)
		x = self.encoder_ReLU_0(x)
		# print(f"Layer 0: {x.shape = }")
		size_0 = x.shape
		x, indices_0 = self.encoder_MaxPool_0(x)
		x = self.encoder_Conv2d_1(x)
		x = self.encoder_ReLU_1(x)
		# print(f"Layer 1: {x.shape = }")
		size_1 = x.shape
		x, indices_1 = self.encoder_MaxPool_1(x)
		size_2 = x.shape
		x = self.encoder_flatten_2(x)
		x = self.encoder_dense_2(x)
		x = self.encoder_dense_3(x)
		return x, ((indices_0, size_0), (indices_1, size_1), size_2)

	def decoder(self, x, pooldata):
		# print(f"Layer 0: {x.shape = }")
		x = self.decoder_dense_m2(x)
		# print(f"Layer 1: {x.shape = }")
		x = self.decoder_dense_m1(x)
		# print(f"Layer 2: {x.shape = }")
		x = x.view(pooldata[2])
		# print(f"decoder begin: {x.shape = }")
		# , output_size=torch.Size([1, 16, 19, 27]))
		x = self.decoder_MaxunPool_0(x, pooldata[1][0], output_size=pooldata[1][1])
		# print(f"decoder unpooling: {x.shape = }")
		x = self.decoder_ConvT2d_0(x)
		x = self.decoder_ReLU_0(x)
		# , output_size=torch.Size([1, 6, 116, 156]))
		x = self.decoder_MaxunPool_1(x, pooldata[0][0], output_size=pooldata[0][1])
		x = self.decoder_ConvT2d_1(x)
		x = self.decoder_ReLU_1(x)
		return x

	def forward(self, x):
		x, pooldata = self.encoder(x)
		# print(f"{x.shape = }")
		x = self.decoder(x, pooldata)
		return x


class PoolingAutoEncoder():
	def __init__(self, config) -> None:
		self.config = config
		self.model_path = f"{self.config.model_dir}{self.config.name}"
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model = AutoEncoderModel().to(self.device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
		self.criterion = nn.MSELoss()
		self.loss = 0.

	def train(self, batch):
		# print(batch.shape)
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
			X, _ = self.model.encoder(X)
			return X

	def load(self):
		self.model.load_state_dict(torch.load(self.model_path))
		self.model.eval()

	def save(self):
		torch.save(self.model.state_dict(), self.model_path)
