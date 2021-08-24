import matplotlib.pyplot as plt
import numpy as np
from Tangents import Tangents



class RoadDeducer():
	def __init__(self) -> None:
		self.p = []
		self.x = []
		self.y = []
		self.z = []
		self.r = []

	def add(self, circle):
		self.p.append(circle)
		self.x.append(circle.x)
		self.y.append(circle.y)
		self.z.append(circle.z)
		self.r.append(circle.r)

	def plot_plot(self):
		plt.plot(self.x, label="x")
		plt.plot(self.y, label="y")
		plt.plot(self.z, label="z")
		plt.plot(self.r, label="r")
		plt.legend(loc="upper left")
		plt.show()

	def get_tangents(self, start, ax):
		a = self.p[start]
		for i in range(start + 1, len(self.p)):
			b = self.p[i]

			t = Tangents(a, b)
			if t.valid():
				t.do()
				self.plot_tangents(ax, start, t)
				return



	def plot_tangents(self, ax, c, t):
		NUM_COLORS = 30
		# cm = plt.get_cmap('gist_rainbow')

		(xa1, ya1), (xa2, ya2) = t.l1
		(xb1, yb1), (xb2, yb2) = t.l2

		# ax.plot(*t.l1, color="salmon", label="tan1")
		# ax.plot(*t.l2, color="firebrick", label="tan2")

		ax.scatter(t.xp, t.yp, color="purple", s=10, label="P")

		ax.scatter(t.a, t.b, color="skyblue", s=10, label="o1")
		ax.scatter(t.c, t.d, color="royalblue", s=10, label="o2")

		min_x = min([t.c0.x, t.c1.x])
		max_x = max([t.c0.x, t.c1.x])

		ccc = 'k' if c % 2 else 'y'
		(x1, y1), (x2, y2) = t.road
		ax.plot((x1, x2), (y1, y2), color=ccc, label="line2")

		ax.scatter((xa1, xa2), (ya1, ya2), color="salmon", label="tan1")
		ax.scatter((xb1, xb2), (yb1, yb2), color="firebrick", label="tan2")

	def plot_circles(self):
		print(len(self.p))
		NUM_COLORS = len(self.p) + 1
		cm = plt.get_cmap('gist_rainbow')
		fig, ax = plt.subplots()

		circles = []
		for i, p in enumerate(self.p):
			c = plt.Circle((p.x, p.y), p.r, color=cm(
				1.*i/NUM_COLORS), fill=False, label=f"c{i}")
			circles.append(c)

			self.get_tangents(i, ax)

		for c in circles:
			ax.add_patch(c)

		ax.set_xlim((-10, 50))
		ax.set_ylim((-10, 50))

		# plt.legend(loc="upper left")
		plt.show()
