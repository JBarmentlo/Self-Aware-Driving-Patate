import numpy as np
from config import config, cte_config

def linear_unbin(arr, turn_bins=config.turn_bins):
	#TODO: remove function
	"""
	Convert a categorical array to value.
	See Also
	--------
	linear_bin
	"""
	# print(arr)
	if not len(arr) == turn_bins:
		raise ValueError(f'Illegal array length, must be {turn_bins}')
	b = np.argmax(arr)
	a = b * (2 / (turn_bins - 1)) - 1
	# print("unbin", a, b)
	return a


def linear_bin(a, turn_bins=config.turn_bins):
	"""
	Convert a value to a categorical array.
	Parameters
	----------
	a : int or float
		A value between -1 and 1
	Returns
	-------
	list of int
		A list of length 15 with one item set to 1, which represents the linear value, and all other items set to 0.
	"""
	# print(a)
	a = a + 1
	b = round(a / (2 / (turn_bins - 1)))
	arr = np.zeros(turn_bins)
	arr[int(b)] = 1
	# print("bin", a, arr)
	return arr

def is_cte_out(cte):
	cte += cte_config.cte_offset
	if abs(cte) > cte_config.max_cte:
		return True
	else:
		return False
