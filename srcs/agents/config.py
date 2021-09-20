class DotDict(dict):
	"""
	a dictionary that supports dot notation 
	as well as dictionary access notation 
	usage: d = DotDict() or d = DotDict({'val1':'first'})
	set attributes: d.val2 = 'second' or d['val2'] = 'second'
	get attributes: d.val2 or d['val2']
	"""
	__getattr__ = dict.__getitem__
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

config = DotDict
config.policy = DotDict
config.Qfunction = DotDict

# General
config.state_shape = (8,)
config.state_shape = (4, 8)
config.num_actions = 2
config.batch_size = 32
config.discount_factor = 0.99
config.lr = 3e-4
config.train_delay_step = 2


# Policy
config.policy.lr = 3e-4
config.policy.state_shape = config.state_shape
config.policy.num_actions = config.num_actions
config.policy.log_std_min = -2
config.policy.log_std_max = 2
config.policy.epsilon = 1e-6
config.policy.alpha = 0.9
config.policy.alpha_lr = 3e-4

# QFunctions
config.Qfunction.state_shape = config.state_shape
config.Qfunction.num_actions = config.num_actions
config.Qfunction.lr = 3e-4
config.Qfunction.tau = 0.01
