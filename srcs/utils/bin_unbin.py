def bin_to_val(bin, bounds, slices):
	out = (bin / (slices - 1)) * (bounds[1] - bounds[0]) + bounds[0]
	return out


def bin_to_val_torch(bin, bounds, slices):
	out = (bin / (slices - 1)) * (bounds[1] - bounds[0]) + bounds[0]
	return out.item()


def val_to_bin(val, bounds, slices):
	out = (val - bounds[0]) / (bounds[1] - bounds[0]) * (slices - 1)
	return out


def val_to_bin_torch(val, bounds, slices):
	out = (val - bounds[0]) / (bounds[1] - bounds[0]) * (slices - 1)
	return out.item()
