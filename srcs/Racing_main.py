import gym
import gym_donkeycar
import uuid

from Simulator import Simulator
from agents import DQNAgent
from config import config_Agent, config
from S3 import S3
from Preprocessing import PreprocessingVannilla
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from skimage.filters import threshold_multiotsu


    # "donkey-warehouse-v0"
    # "donkey-generated-roads-v0"
    # "donkey-avc-sparkfun-v0"
    # "donkey-generated-track-v0"
    # "donkey-roboracingleague-track-v0"
    # "donkey-waveshare-v0"
    # "donkey-minimonaco-track-v0"
    # "donkey-warren-track-v0"
    # "donkey-thunderhill-track-v0"
    # "donkey-circuit-launch-track-v0"


host = "donkey-sim.roboticist.dev" 
env_name = "donkey-circuit-launch-track-v0"
config_Simulator = {"exe_path": "manual",
						"host": host,
						"body_style": "donkey",
						"body_rgb": (128, 128, 128),
						"car_name": "42AI Potato Qarnot",
						"font_size": 100,
						"racer_name": "DDQN",
						"country": "FR",
						"bio": "Learning to drive w DDQN RL",
						"guid": str(uuid.uuid4()),
						"max_cte": 10,
				}

if __name__ == "__main__":
	env = gym.make(env_name, conf=config_Simulator)
	
	S3 = S3(config.config_NeuralPlayer.config_Datasets.S3_bucket_name)
	agent = DQNAgent(config=config_Agent, S3=S3)
	agent.config.epsilon = 0.1
	preprocessor = PreprocessingVannilla(config.config_NeuralPlayer.config_Preprocessing)
	
	env.reset()
	i = 0
	state, reward, done, infos = env.step([0, 0.1])
	while(i < 1000):
		processed_state = preprocessor.process(state)
		action = agent.get_action(processed_state)
		state, reward, done, infos = env.step(action)
		print(action, done, infos)
		i += 1
