from simlaunch3000 import Client, start_server
import time


if __name__ == "__main__":
	c = Client()
	c.request_simulator()
	time.sleep(6)
	c.kill_sim()