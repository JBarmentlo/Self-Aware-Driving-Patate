import numpy as np
from config import config, cte_config
import pickle
import json
from datetime import datetime
from s3 import S3


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


def upload_pickle_file(file_name, content):
	with open(file_name, "wb") as f:
		pickle.dump(content, f)


def upload_json_file(file_name, content):
	with open(file_name, "w") as f:
		json.dump(content, f)


def read_json_file(name):
	with open(name, "r") as f:
		result = json.load(f)
	return(result)


def read_pickle_file(name):
	with open(name, "rb") as f:
		result = pickle.load(f)
	return(result)


def append_db(episode_memory, state, action, reward, new_state, done, info):
		### TODO: rajouter check des arguments
	# MEMORY for database (to train without simulator)
	episode_memory.append((state, action, reward, new_state, done, info))
	

def save_memory_db(memory_list, infos, episode, our_s3 = None):
	file_name = f"{infos['prefix']}_{episode}{config.memory_sufix}"
	if our_s3 != None:
		our_s3.pickle_upload(file_name, memory_list)
	else:
		upload_pickle_file(file_name, memory_list)


def init_dic_info(args, our_s3 = None): ### TODO add infos about last commit ec...
	date = datetime.now().strftime("%d_%m_%Hh%Mm")
	name = config.name_neural_player
	if args.supervised:
		name = config.name_human_player
	if args.destination == "s3":
		folder = config.s3_memory_folder
	else:
		folder = config.local_memory_folder
	info_prefix = f"{folder}/{name}_{date}"
	infos = {"name" : name, "date" : str(date), "env_name" : args.env_name, "prefix" : info_prefix}
	info_file_name = f"{info_prefix}{config.info_sufix}"
	if args.destination == "s3" and our_s3:
		our_s3.upload_json_file(info_file_name, infos)
	else:
		upload_json_file(info_file_name, infos)
	return (infos)

