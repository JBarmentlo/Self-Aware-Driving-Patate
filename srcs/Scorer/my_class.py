import matplotlib.pyplot as plt


class Point():
	def __init__(self, e) -> None:
		self.x = e["pos"][0]
		self.y = e["pos"][1]
		self.z = e["pos"][2]
		self.r = e["cte"]


class Line():
	def __init__(self) -> None:
		self.p = []
		self.x = []
		self.y = []
		self.z = []
		self.r = []

	def add(self, point):
		self.p.append(point)
		self.x.append(point.x)
		self.y.append(point.y)
		self.z.append(point.z)
		self.r.append(point.r)

	def plot_plot(self):
		plt.plot(self.x, label="x")
		plt.plot(self.y, label="y")
		plt.plot(self.z, label="z")
		plt.plot(self.r, label="r")

		plt.show()
