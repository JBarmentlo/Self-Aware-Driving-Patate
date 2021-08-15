from os import path
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from AutoEncoderDummy import UndercompleteAutoEncoder
from ContractiveAutoEncoder import ContractiveAutoEncoder
from PoolingAutoEncoder import PoolingAutoEncoder


PREPROCESSING = False

def show_imgs(X, Y, path=None):
	m = len(X)
	_, axs = plt.subplots(2, m)
	for i in range(m):
		print(i)
		to_plot = X[i].cpu().detach().numpy()
		to_plot = np.moveaxis(to_plot, 0, -1)
		axs[0, i].imshow(to_plot, interpolation='nearest')
		to_plot = Y[i].cpu().detach().numpy()
		to_plot = np.moveaxis(to_plot, 0, -1)
		axs[1, i].imshow(to_plot, interpolation='nearest')
	if path:
		plt.savefig(path)
	plt.show()

def visu_plot(ae, b, save=None):
	X = b[0]
	Y = b[0]
	print()
	if PREPROCESSING:
		X = preprocessing(X)
		Y = preprocessing(Y)
	X = ae.predict(X)
	show_imgs(X, Y, path=save)


def load(data_dir="simulator_cache"):
	transform = transforms.Compose([transforms.ToTensor()])
	dataset = datasets.ImageFolder(data_dir, transform=transform)
	train = torch.utils.data.DataLoader(dataset, batch_size=64,
                                          num_workers=1,
                                          pin_memory=True, shuffle=True)
	test = torch.utils.data.DataLoader(dataset, batch_size=4,
                                    num_workers=1,
                                    pin_memory=True, shuffle=True)
	return train, test


def preprocessing(data, simplify=30.):
	# return data
	data = torch.div(data * 255., simplify, rounding_mode="trunc") * simplify / 255
	# data = (data // simplify).type(torch.float32) * simplify
	return data


if __name__ == "__main__":
	train, test = load()

	# type_ = "uae"
	type_ = "pae"
	# type_ = "cae"
	if type_ == "uae":
		ae = UndercompleteAutoEncoder(1, 10, learning_rate=1e-3)
	elif type_ == "cae":
		ae = ContractiveAutoEncoder(1, 10, learning_rate=1e-3)
	elif type_ == "pae":
		ae = PoolingAutoEncoder(1, 10, learning_rate=1e-3)

	i = 0
	m = len(train)
	nb_epochs = 15
	for e in range(nb_epochs):
		print(f"Epoch {e}/{nb_epochs}")
		for batch in train:
			data = batch[0]
			if PREPROCESSING:
				data = preprocessing(data)
			ae.loss = 0.
			ae.train(data)
			i += 1

	for batch in test:
		visu_plot(ae, batch, save=f"srcs/AutoEncoder/results/{type_}.png")
		break
