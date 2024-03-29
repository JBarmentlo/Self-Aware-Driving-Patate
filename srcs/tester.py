import matplotlib.pyplot as plt
from S3 import S3
import io

from config import config
from HumanPlayer import HumanPlayer
from Simulator import Simulator
from utils import free_all_sims

# plt.figure()
# plt.plot([1, 2])
# plt.title("test")
# # plt.savefig("hey")

# my_s3 = S3("deyopotato")
# buf = io.BytesIO()
# plt.savefig(buf, format='png')
# buf.seek(0)
# my_s3.upload_bytes(buf, "model_cache/autoencoder/images_results/weshwesh")
free_all_sims(1)
simulator = Simulator(config.config_Simulator, "donkey-circuit-launch-track-v0")
human = HumanPlayer(config.config_HumanPlayer, env = simulator.env, simulator = simulator)
human.do_race()