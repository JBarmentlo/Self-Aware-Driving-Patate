# import matplotlib.pyplot as plt
# from S3 import S3
# import io

# from config import config
# from HumanPlayer import HumanPlayer
# from Simulator import Simulator


# plt.figure()
# plt.plot([1, 2])
# plt.title("test")
# # plt.savefig("hey")

# my_s3 = S3("deyopotato")
# buf = io.BytesIO()
# plt.savefig(buf, format='png')
# buf.seek(0)
# my_s3.upload_bytes(buf, "model_cache/autoencoder/images_results/weshwesh")

# simulator = Simulator(config.config_Simulator, "donkey-generated-roads-v0")
# human = HumanPlayer(config.config_HumanPlayer, env = simulator.env, simulator = simulator)
# human.do_race()

import pandas as pd


# Logger = logging.getLogger("Central Agent Master")
# Logger.setLevel(logging.INFO)
# stream = logging.StreamHandler()
# Logger.addHandler(stream)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class ScoreDataGatherer():
	'''
		Container for the scores, speeds and distances of the races,
	'''
	def __init__(self):
		self.data		: pd.DataFrame = pd.DataFrame(columns=["score", "speed", "dist", "episode", "epsilon"])
		self.eval_data	: pd.DataFrame = pd.DataFrame(columns=["score", "speed", "dist", "episode", "epsilon"])
		plt.ion()
		plt.show()


	def add_point(self, score, speed, dist, episode, epsilon):
		self.data = self.data.append({"score" : score, "speed" : speed, "dist" : dist, "episode" : episode, "epsilon" : epsilon}, ignore_index= True)


	def add_eval_point(self, score, speed, dist, episode, epsilon):
		self.eval_data = self.eval_data.append({"score" : score, "speed" : speed, "dist" : dist, "episode" : episode, "epsilon" : epsilon}, ignore_index= True)


	def plot(self):
		'''
			will hang until closed
		'''
		ax = self.data.plot(kind = "scatter", x = "episode", y = "score", color = "C3", label = "Score")
		self.data.plot(kind = "scatter", x = "episode", y = "speed", color = "C2", label = "Speed", ax = ax)
		self.data.plot(kind = "scatter", x = "episode", y = "dist", color = "C4", label = "Distance", ax = ax)
		plt.draw()
		plt.pause(0.001)

s = ScoreDataGatherer()
s.add_point(1,2,3,4,5)
s.add_point(2,3,4,5,6)
s.add_point(3,4,5,6,7)
s.add_point(4,5,6,7,8)