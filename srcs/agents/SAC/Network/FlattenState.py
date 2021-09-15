from torch import nn
import torch.nn.functional as F

class FlattenState(nn.Module):
	def __init__(self, input_channels):
		super(FlattenState, self).__init__()
		self.conv0 = nn.Conv1d(input_channels, 4, kernel_size=1, stride=1, padding=0)
		self.conv1 = nn.Conv1d(4, 2, kernel_size=1, stride=1, padding=0)
		self.conv2 = nn.Conv1d(2, 1, kernel_size=1, stride=1, padding=0)

		self.flatten = nn.Flatten()

	def forward(self, x):
		x = F.relu(self.conv0(x))
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = self.flatten(x)
		return x