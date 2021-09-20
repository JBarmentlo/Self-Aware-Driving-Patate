from ..utils import DotDict

net_config = DotDict()
net_config.host = 'localhost'
net_config.start_port = 9091

net_config.server_port = 9083
net_config.server_refresh_time = 1 # seconds to wait before listening again

net_config.start_sim_request = "start"
net_config.ping_request = "ping"
net_config.kill_request = "kill"
net_config.stop_using_sim_request = "release"



# * The Host option remains unused for now