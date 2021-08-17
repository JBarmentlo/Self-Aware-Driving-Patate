import torch

def bin_to_val(bin, bounds, slices):
	if (slices == 1):
		out = torch.ones(bin.size(), dtype = torch.float32) * bounds[0]
	else:
		out = (bin / (slices - 1)) * (bounds[1] - bounds[0]) + bounds[0]
	return out


def bin_to_val_torch(bin, bounds, slices):
	if (slices == 1):
		out = torch.ones(bin.size(), dtype = torch.float32) * bounds[0]
	else:
		out = (bin / (slices - 1)) * (bounds[1] - bounds[0]) + bounds[0]
	return out.item()


def val_to_bin(val, bounds, slices):
	if (slices == 1):
		out = torch.zeros(val.size(), dtype = torch.int64)
	else:
		out = (val - bounds[0]) / (bounds[1] - bounds[0]) * (slices - 1)
		out.to(torch.int64)
	return out


def val_to_bin_torch(val, bounds, slices):
	if (slices == 1):
		out = torch.zeros(val.size(), dtype = torch.int64)
	else:
		out = (val - bounds[0]) / (bounds[1] - bounds[0]) * (slices - 1)
		out.to(torch.int64)
	return out.item()
