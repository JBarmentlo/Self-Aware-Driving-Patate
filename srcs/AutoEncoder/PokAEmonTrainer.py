import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

def plot_comparaison(X, Y, path=None, plot=False):
	m = len(X)
	_, axs = plt.subplots(2, m)
	for i in range(m):
		to_plot = X[i].cpu().detach().numpy()
		to_plot = np.moveaxis(to_plot, 0, -1)
		axs[0, i].imshow(to_plot, interpolation='nearest')
		to_plot = Y[i].cpu().detach().numpy()
		to_plot = np.moveaxis(to_plot, 0, -1)
		axs[1, i].imshow(to_plot, interpolation='nearest')
	if path:
		plt.savefig(path)
	if plot:
		plt.show()

class AutoEncoderTrainer():
	def __init__(self, AutoEncoder, config, plot=False) -> None:
		self.config = config
		self.ae = AutoEncoder
		self.plot = plot
		self.load()
		self.train()

	def load(self):
		transform = transforms.Compose([transforms.ToTensor()])
		self.dataset = datasets.ImageFolder(self.config.train_dir, transform=transform)
		self.train_dataset = torch.utils.data.DataLoader(self.dataset, batch_size=self.config.batch_size,
										num_workers=1,
										pin_memory=True, shuffle=True)
		self.test_dataset = torch.utils.data.DataLoader(self.dataset, batch_size=4,
										num_workers=1,
										pin_memory=True, shuffle=True)
	
	def train(self):
		epochs = self.config.epochs
		for e in range(epochs):
			print(f"Epoch {e}/{epochs}")
			for batch in self.train_dataset:
				data = batch[0]
				self.ae.loss = 0.
				self.ae.train(data)
		self.ae.save()
		self.test()

	def test(self):
		result_plot = f"srcs/AutoEncoder/results/{self.config.name}_{self.ae.__class__.__name__}.png"
		for batch in self.test_dataset:
			X = batch[0]
			Y = batch[0]
			Y_ = self.ae.predict(X)
			plot_comparaison(Y_, Y, path=result_plot, plot=self.plot)
			break