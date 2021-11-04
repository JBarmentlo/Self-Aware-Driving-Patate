from typing import List
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class DistScorer():
	'''
		Simple distance scorer, approximates track to be linear (valid because short).
		the prevlastpos attribute could be used to distiguish forward from backwards motion
	'''
	def __init__(self):
		self.prevlastpos 		: np.ndarray
		self.lastpos 			: np.ndarray
		self.last_cte 			: float
		self.distances			= []
		self.speeds				= []
		self.times  			= []
		self.total_frames 		= []
		self.scores 			= []
		self.current_race_dist  : float
		self.current_race_time 	: float
		self.start_time 		: datetime


	def first_point(self, info : dict):
		self.lastpos = np.array(info["pos"])

		self.prevlastpos = np.array(info["pos"])

		self.last_cte = info["cte"]

		self.current_race_dist = 0
		self.start_time = datetime.now()


	def add_point(self, info : dict):
		newpos = np.array(info["pos"])
		new_cte = info["cte"]
		cte_diff = abs(new_cte - self.last_cte)
		pos_diff = newpos - self.lastpos
		pos_diff_norm = np.linalg.norm(pos_diff)
		dist = np.sqrt(abs(pos_diff_norm ** 2 - cte_diff ** 2))
		if (np.linalg.norm(newpos - self.lastpos) > np.linalg.norm(newpos - self.prevlastpos)):
			dist = -1 * dist
		self.current_race_dist += dist
		self.last_cte = new_cte
		self.prevlastpos = self.lastpos
		self.lastpos = newpos


	def end_race(self, total_frames):
		self.current_race_time = (datetime.now() - self.start_time).total_seconds()

		self.distances.append(self.current_race_dist)
		
		self.total_frames.append(total_frames)

		self.times.append(self.current_race_time)

		self.scores.append(self.get_current_race_score())

		self.speeds.append(self.get_current_race_speed())


	def get_current_race_time(self):
		return (datetime.now() - self.start_time).total_seconds()


	def get_current_race_score(self):
		return (self.current_race_dist / np.sqrt(self.get_current_race_time()))


	def get_current_race_dist(self):
		return (self.current_race_dist)

	
	def get_current_race_speed(self):
		return (self.current_race_dist / self.get_current_race_time())

		
	def plot(self):
		plt.plot(self.total_frames, self.distances)
		plt.label("Scores per frame")
		plt.show()
	



		
