from inputs import get_key
from utils import init_dic_info, append_db, save_memory_db
from Simulator import Simulator
from config import config
import threading
from s3 import S3

class HumanPlayer():
    def __init__(self, args):
        self.args = args
        self.our_s3 = None
        if self.args.destination == "s3":
              self.our_s3 = S3()
        self.general_infos = init_dic_info(self.args, self.our_s3)
        Simulator(self)
        self.episode_memory = []
        self.db = None
        self.db_len = 0
        self.episode_memory = []
        self.throttle, self.steering, self.stop = 0, 0, 0
        self.commands = self.throttle, self.steering, self.stop
        self.run_supervised()
        # except KeyboardInterrupt:
        #     print("stopping run...")
        # finally:
        #     self.env.unwrapped.close()
    
    def run_supervised(self):
        print("-------- PRESS any key to start connecting the keyboard, it can take a while...")
        state = self.env.reset()
        get_key()
        print("\n\n**********         Now you can start driving with your KEYPADS :) :)         **********\n\n")
        while self.stop == 0:
            commands = self.commands
            self.get_command()
            if self.commands != commands:
                action = [self.steering, self.throttle]
                new_state, reward, done, info = self.env.step(action)
            else:
                new_state, reward, done, info = self.env.viewer.observe()
            if threading.active_count() <= config.max_threads:
                t = threading.Thread(target=append_db, args=[self.episode_memory, state, action, reward, new_state, done, info])
                t.start()
            state = new_state
        print("stopping")
        save_memory_db(self.episode_memory, self.general_infos, 0, self.our_s3)
        
    
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