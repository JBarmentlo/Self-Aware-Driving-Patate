from simlaunch3000 import Client
from simlaunch3000.src.config import net_config, sim_config

def free_all_sims(number_of_sims = None):
	'''
		Frees all sims quite brutally. If no argument is given frees all possible sims.
		Else frees nmumber_of_sims starting from the first port in simlaunch3000 config (9091)
	'''
	c = Client()
	if number_of_sims is None:
		number_of_sims = sim_config.max_concurrent_sims
	for port in range(net_config.start_port, net_config.start_port + number_of_sims):
		c.sim_port = port
		try:
			c.release_sim()
		except:
			print("Normal error, brutally releasing all sims. Even unexisting ones")