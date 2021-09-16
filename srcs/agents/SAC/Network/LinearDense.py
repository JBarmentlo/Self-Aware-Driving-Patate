from torch import nn
import torch.nn.functional as F
import numpy as np

class LinearDense(nn.Module):
	def __init__(self, input_size, output_size, layers):
		super(LinearDense, self).__init__()
		self.layers = nn.ModuleList()
		self._build(input_size, output_size, layers)

	def _build(self, input_, output, size):
		len_list = np.linspace(input_, output, size + 1).astype(int)
		for n_t, n_t_1 in zip(len_list[:-1], len_list[1:]):
			if len(self.layers):
				self.layers.append(nn.ReLU())
			self.layers.append(nn.Linear(n_t, n_t_1))

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
		return x


