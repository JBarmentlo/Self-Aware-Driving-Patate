import numpy as np

def to_numpy_32(list_items):
	new_list = []
	for item in list_items:
		item = np.array(item, dtype=np.float32)
		new_list.append(item)
	return(new_list)