from inputs import get_key
import threading
import torch
import logging

from RewardOpti import RewardOpti
from agents.Agent import DQNAgent
from Preprocessing import Preprocessing
from Simulator import Simulator
import utils
from S3 import S3
from SimCache import SimCache

Logger = logging.getLogger("HumanPlayer")
Logger.setLevel(logging.INFO)
stream = logging.StreamHandler()
Logger.addHandler(stream)

class  HumanAgent():
    def __init__(self, config):
        self.config = config
        self.conf_data, self.conf_s3 = config.config_Datasets, config.config_Datasets.config_S3
        self._init_S3(self.conf_s3)
        self.SimCache = SimCache(self.conf_data, self.S3)


    def _init_S3(self, config):
        self.S3 = None
        if self.conf_data.S3_connection == True:
            self.S3 = S3(config)
    
    
    def add_simcache_point(self, datapoint):
        if self.conf_data.save_SimCache == True:
            if self.SimCache.datapoints_counter + 1 > self.conf_data.size_SimCache:
                self.SimCache.upload()
            self.SimCache.add_point(datapoint)


    def get_action(self, state, episode = 0):
        pass




class HumanPlayer():
	def __init__(self, config, env, simulator):
		self.config = config
		self.env = env
		self.simulator = simulator
		self.stop, self.throttle, self.steering = 0, 0, 0
		self.commands = self.throttle, self.steering, self.stop


	def append_memory(self, state, action, reward, new_state, done, info):
		self.memory.append([state, action, reward, new_state, done, info])


	def save_memory(self, file_name):
		with open(file_name, "wb") as f:
			pickle.dumps(self.memory, f)


	def do_races(self):
		Logger.info(f"Starting human race.")
		self.simulator = utils.fix_cte(self.simulator)
		self.env = self.simulator.env
		state, reward, done, infos = self.env.step([self.steering, self.throttle])
		Logger.info(f"Cte is fixed you can now start your race by pressing any key.")
		while self.stop == 0:
			commands = self.commands
			self.get_command()
			if self.commands != commands:
				action = [self.steering, self.throttle]
				new_state, reward, done, info = self.env.step(action)
			else:
				new_state, reward, done, info = self.env.viewer.observe()
			if threading.active_count() <= config.max_threads:
				t = threading.Thread(target=self.append_memory, args=[self.memory, state, action, reward, new_state, done, info])
				t.start()
			state = new_state
		print("stopping")
		save_memory("weshwesh.human")
		return

	
	def get_command(self):
		event = get_key()[1]
		if event.code == "KEY_ESC" and event.state == 1:
			self.stop = 1
		elif event.code == "KEY_UP" and event.state == 1:
			self.throttle = abs(self.throttle * config.coef)
		elif (event.code == "KEY_UP" or event.code == "KEY_DOWN") and event.state == 0:
			self.throttle = config.init_throttle
		elif event.code == "KEY_DOWN" and event.state == 1:
			self.throttle = abs(self.throttle * config.coef) * -1
		elif event.code == "KEY_LEFT" and event.state == 1:
			if self.steering == 0:
				self.steering = config.init_steering * -1
			else:
				self.steering = abs(self.steering * config.coef) * -1
		elif (event.code == "KEY_LEFT" or event.code == "KEY_RIGHT") and event.state == 0:
			self.steering = 0
		elif event.code == "KEY_RIGHT" and event.state == 1:
			if self.steering == 0:
				self.steering = config.init_steering
			else:
				self.steering = abs(self.steering * config.coef)
		self.check_max_min()
		self.commands = self.stop, self.throttle, self.steering 


	def check_max_min(self):
		if self.throttle > config.max_throttle:
			self.throttle = config.max_throttle
		if self.throttle < config.min_throttle:
			self.throttle = config.min_throttle
		if self.steering > config.max_steering:
			self.steering = config.max_steering
		if self.steering < config.min_steering:
			self.steering = config.min_steering
	
=======
    def __init__(self, config, env, simulator):
        self.config = config
        self.env = env
        self.agent =  HumanAgent(config.config_Agent)
        self.simulator = simulator


    def do_races(self):
        Logger.info(f"Starting human race.")
        # self.simulator = utils.fix_cte(self.simulator)
        self.env = self.simulator.env
        self.stop, self.throttle, self.steering = 0, 0, 0
        self.commands = self.stop, self.throttle, self.steering
        state, reward, done, infos = self.env.step([self.steering, self.throttle])
        Logger.info(f"Cte is fixed you can now start your race by pressing any key.")
        while self.stop == 0:
            commands = self.commands
            self.get_command()
            if self.commands != commands:
                action = [self.steering, self.throttle]
                print(f"Getting action: {action}")
                new_state, reward, done, infos = self.env.step(action)
            else:
                print(f"observing")
                new_state, reward, done, infos = self.env.step(action) # self.env.viewer.observe()
            self.agent.add_simcache_point([state, action, new_state, reward, done, infos])
            # if threading.active_count() <= config.max_threads:
                # t = threading.Thread(target=self.agent.add_simcache_point, args=[state, action, reward, new_state, done, info])
                # t.start()
            state = new_state
        print("stopping")
        self.agent.SimCache.upload()
        return

    
    def get_command(self):
        event = get_key()[1]
        print(f"{event.state = }")
        if event.code == "KEY_ESC" and event.state == 1:
            self.stop = 1
        # elif event.code == "KEY_UP" and event.state == 1:
        #     self.throttle = abs(self.throttle * config.coef)
        # elif (event.code == "KEY_UP" or event.code == "KEY_DOWN") and event.state == 0:
        #     self.throttle = config.init_throttle
        # elif event.code == "KEY_DOWN" and event.state == 1:
        #     self.throttle = abs(self.throttle * config.coef) * -1
        # elif event.code == "KEY_LEFT" and event.state == 1:
        #     if self.steering == 0:
        #         self.steering = config.init_steering * -1
        #     else:
        #         self.steering = abs(self.steering * config.coef) * -1
        # elif (event.code == "KEY_LEFT" or event.code == "KEY_RIGHT") and event.state == 0:
        #     self.steering = 0
        # elif event.code == "KEY_RIGHT" and event.state == 1:
        #     if self.steering == 0:
        #         self.steering = config.init_steering
        #     else:
        #         self.steering = abs(self.steering * config.coef)
        # self.check_max_min()
        self.commands = self.stop, self.throttle, self.steering 


    # def check_max_min(self):
    #     if self.throttle > config.max_throttle:
    #         self.throttle = config.max_throttle
    #     if self.throttle < config.min_throttle:
    #         self.throttle = config.min_throttle
    #     if self.steering > config.max_steering:
    #         self.steering = config.max_steering
    #     if self.steering < config.min_steering:
    #         self.steering = config.min_steering
 