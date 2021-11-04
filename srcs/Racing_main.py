import gym
import gym_donkeycar
import uuid

from Simulator import Simulator
from agents import DQNAgent
from config import config_Agent, config

host = "donkey-sim.roboticist.dev" 
env_name = "donkey-generated-roads-v0"
config_Simulator = {"exe_path": "manual",
						"host": host,
						"body_style": "donkey",
						"body_rgb": (128, 128, 128),
						"car_name": "PATATOOOO",
						"font_size": 100,
						"racer_name": "DDQN",
						"country": "FR",
						"bio": "Learning to drive w DDQN RL",
						"guid": str(uuid.uuid4()),
						"max_cte": 10,
				}

if __name__ == "__main__":
	env = gym.make(env_name, conf=config_Simulator)
	env.reset()
	i = 0
	while(i < 1000):
		env.step([1.0,1.0])
		i += 1