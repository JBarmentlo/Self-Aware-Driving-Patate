from ..utils import DotDict


sim_config = DotDict()
sim_config.max_concurrent_sims = 20 
sim_config.time_till_timeout = 600
sim_config.kill_on_timeout = False