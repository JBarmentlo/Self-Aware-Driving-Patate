import io
import torch
import pickle

from Simulator import Simulator
from NeuralPlayer import NeuralPlayer
from HumanPlayer import HumanPlayer
from S3 import S3

from config import config

config_S3 = config.config_NeuralPlayer.config_Agent.config_S3
config_Agent = config.config_NeuralPlayer.config_Agent

neural = NeuralPlayer(config.config_NeuralPlayer, None, None)
agent = neural.agent

# config_Agent.S3_connection = True 

# config_Agent.model_to_load      = f"model_cache/dedequene.modelo.2500"

# agent._load_model(config_Agent.model_to_load)

# config_Agent.model_to_load      = f"/Users/deyaberger/projects/patata/model_cache/dedequene.modelo.2500"


# model_st = agent.model.state_dict()
# buffer = io.BytesIO()
# torch.save(model_st, buffer)
# myS3 = S3(config_S3)
# test = pickle.dumps(model_st)
# path = "/Users/deyaberger/projects/patata/model_cache/dedequene.modelo.2500"
# with open(path, "rb") as f:
#     buf = io.BytesIO(f.read())
# test = myS3.get_bytes(config.config_NeuralPlayer.config_Agent.model_to_load)
# test = io.BytesIO(test)
# agent.model.load_state_dict(torch.load(test, map_location=torch.device('cpu')))