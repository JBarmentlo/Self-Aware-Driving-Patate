import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from Memory import AutoEncoderDataset

def plot_comparaison(X, Y, path=None, plot=False):
	m = len(X)
	if m <= 1:
		return
	if not path and not plot:
		return
	if not hasattr(plot_comparaison, "axs"):
		plt.ion()
		_, axs = plt.subplots(2, m)
		plot_comparaison.axs = axs
	DataPrep = lambda X: np.moveaxis((X.cpu().detach().numpy() * 255).astype(np.uint8), 0, -1)
	for i in range(m):
		plot_comparaison.axs[0, i].imshow(DataPrep(X[i]), interpolation='nearest')
		plot_comparaison.axs[1, i].imshow(DataPrep(Y[i]), interpolation='nearest')
	if path:
		plt.savefig(path)
	if plot:
		plt.show()
		plt.pause(0.001)

class AutoEncoderTrainer():
	def __init__(self, AutoEncoder, config, plot=False, SimCache=None, Prepocessing=None) -> None:
		print("Init AE")
		self.config = config
		self.ae = AutoEncoder
		self.plot = plot
		self.SimCache = SimCache
		self.Prepocessing = Prepocessing
		self.load()
		self.train()
		print("End AE")

	def _train_test_split(self, dataset, validation_split=0.05):
		dataset_size = len(dataset)
		indices = list(range(dataset_size))
		split = int(np.floor(validation_split * dataset_size))
		self.train_dataset_size = dataset_size - split
		self.test_dataset_size = split
		if True:
			np.random.seed(42)
			np.random.shuffle(indices)
		train_indices, val_indices = indices[split:], indices[:split]

		# Creating PT data samplers and loaders:
		train_sampler = SubsetRandomSampler(train_indices)
		valid_sampler = SubsetRandomSampler(val_indices)

		self.train_dataset = torch.utils.data.DataLoader(dataset, batch_size=self.config.batch_size,
												   sampler=train_sampler)
		self.test_dataset = torch.utils.data.DataLoader(dataset, batch_size=4,
												  sampler=valid_sampler)
		print(f"Train len: {self.train_dataset_size}")
		print(f"Test  len: {self.test_dataset_size}")
		return self.train_dataset, self.test_dataset

	def _load_local(self):
		transform = transforms.Compose([transforms.Resize((120,120), transforms.InterpolationMode.BICUBIC),
										transforms.ToTensor()])
		self.dataset = datasets.ImageFolder(self.config.train_dir, transform=transform)
		# dataset = torch.utils.data.DataLoader(self.dataset, batch_size=self.config.batch_size,
		# 								num_workers=1,
		# 								pin_memory=True, shuffle=True)
		self._train_test_split(dataset, validation_split=0.05)
		
	def _load_s3(self):
		path = self.SimCache.folder + self.SimCache.list_files[self.SimCache.loading_counter]
		self.SimCache.load(path)
		# X_s = []
		dataset = AutoEncoderDataset()
		for datapoint in self.SimCache.data:
			state, action, new_state, reward, done, infos = datapoint
			print(f"{state.shape = }")
			prep_state = self.Prepocessing.before_AutoEncoder(state, training=True)
			print(f"{prep_state.shape = }")
			complement = torch.tensor([infos["cte"]])
			dataset.add((prep_state, complement))
		self.dataset = dataset
		return dataset
	
	def load(self):
		if self.config.config_AE_Datasets.S3_connection:
			dataset = self._load_s3()
		else:
			dataset = self._load_local()
		self._train_test_split(dataset)

	def train(self):
		epochs = self.config.epochs
		for e in range(epochs):
			print(f"Epoch {e}/{epochs}")
			for batch in tqdm(self.train_dataset):
				data = batch[0]
				self.ae.loss = 0.
				self.ae.train(data)
		self.ae.save()
		self.test()

	def test(self):
		result_plot = f"srcs/AutoEncoder/results/{self.config.name}_{self.ae.__class__.__name__}.png"
		total_loss = 0.
		for batch in tqdm(self.test_dataset):
			X = batch[0]
			Y = batch[0]
			Y_ = self.ae.predict(X)
			loss = self.ae.criterion(Y_, Y.to(self.ae.device))
			total_loss += loss.item()
			plot_comparaison(Y_, Y, path=result_plot, plot=self.plot)
			# break
		mean_loss = total_loss / self.train_dataset_size
		print(f"AutoEncoder mean loss is {mean_loss}")
