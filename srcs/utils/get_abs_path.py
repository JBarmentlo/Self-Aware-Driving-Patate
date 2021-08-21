import pathlib

# IF THIS FILE IS MOVED THE FUNCTION WILL BE BROKEN
# If help is needed to adapt the function contact ezalos (on github) or ldevelle (on slack)


def get_path_to_cache(path_complement="model_cache/"):
	"""
		Returns: this function returns the absolute path of this project
		For example on my laptop it is: /home/USER/42/Self-Aware-Driving-Patate/ + path_complement
	"""
	res = pathlib.Path(__file__).parent.parent.parent.absolute()
	res = str(res) + "/" + path_complement
	return res

if __name__ == "__main__":
	res = get_path_to_cache()
	print(res)
