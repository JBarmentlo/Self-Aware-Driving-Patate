import signal
import tensorflow as tf
import gym
import uuid
import gym_donkeycar ## Keep this module 
from utils import save_memory_db
import sys
from tensorflow.compat.v1.keras import backend as K
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Simulator:
    def __init__(self, player):
        # Sim config
        # only needed if TF==1.13.1
        player.sim_config = tf.compat.v1.ConfigProto(log_device_placement=True)
        player.sim_config.gpu_options.allow_growth = True
        print(player.sim_config)

        # Keras session init
        player.sess = tf.compat.v1.Session(config=player.sim_config)
        K.set_session(player.sess)

        # Create env
        player.conf = {"exe_path": player.args.sim,
                        "host": "127.0.0.1",
                        "port": player.args.port,
                        "body_style": "donkey",
                        "body_rgb": (128, 128, 128),
                        "car_name": "me",
                        "font_size": 100,
                        "racer_name": "DDQN",
                        "country": "FR",
                        "bio": "Learning to drive w DDQN RL",
                        "guid": str(uuid.uuid4()),
                        "max_cte": 10,
                }
        player.env = gym.make(
                player.args.env_name, conf=player.conf)
        # Signal handler
        # not working on windows...
        def signal_handler(signal, frame):
                print("catching ctrl+c")
                if player.args.save or player.args.supervised:
                    save_memory_db(player.episode_memory, player.general_infos, "last")
                player.env.unwrapped.close()
                sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGABRT, signal_handler)