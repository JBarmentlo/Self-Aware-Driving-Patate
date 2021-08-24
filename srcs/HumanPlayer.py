from inputs import get_key
import threading
import torch
import logging

from RewardOpti import RewardOpti
from agents.Agent import DQNAgent
from Preprocessing import Preprocessing
from Simulator import Simulator
import utils

Logger = logging.getLogger("HumanPlayer")
Logger.setLevel(logging.INFO)
stream = logging.StreamHandler()
Logger.addHandler(stream)


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
    