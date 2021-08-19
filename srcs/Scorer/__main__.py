import json
import matplotlib.pyplot as plt
import numpy as np

class Point():
	def __init__(self, x, y, z, r) -> None:
		self.x = x
		self.y = y
		self.z = z
		self.r = r

class Tangent():
	# http://www.ambrsoft.com/TrigoCalc/Circles2/Circles2Tangent_.htm

	def __init__(self, ca, cb) -> None:
		if ca.r > cb.r:
			self.c0 = ca
			self.c1 = cb
		else:
			self.c1 = ca
			self.c0 = cb

	def circle_dist(self):
		a = self.c0.x
		c = self.c1.x
		b = self.c0.y
		d = self.c1.y
		D = ((c - a)**2 + (d - b)**2)**(1/2)
		self.D = D
		self.a = a
		self.b = b
		self.c = c
		self.d = d

	def tangent_lines_intersection(self):
		a = self.c0.x
		b = self.c0.y
		r0 = self.c0.r
		d = self.c1.y
		c = self.c1.x
		r1 = self.c1.r

		print(f"xp = ({c = } * {r0 = } - {a = } * {r1 = }) / ({r0 = } - {r1 = })")

		xp = (c * r0 - a * r1) / (r0 - r1)
		yp = (d * r0 - b * r1) / (r0 - r1)

		self.xp = xp
		self.yp = yp
		print(f"{xp = }")
		print(f"{yp = }")

	def tangents_points_r0(self):
		e = 1e-20
		a = self.c0.x
		b = self.c0.y
		r0 = self.c0.r
		xp = self.xp
		yp = self.yp

		l_up = r0**2 * (xp - a)
		r_up = r0 * (yp - b) * ((xp - a)**2 + (yp - b)**2 - r0**2)**(1/2)
		down = (xp - a)**2 + (yp - b)**2

		xt1 = ((l_up + r_up) / down) + a
		xt2 = ((l_up - r_up) / down) + a

		l_up = r0**2 * (yp - b)
		r_up = r0 * (xp - a) * ((xp - a)**2 + (yp - b)**2 - r0**2)**(1/2)
		down = (xp - a)**2 + (yp - b)**2

		yt1 = ((l_up - r_up) / down) + b
		yt2 = ((l_up + r_up) / down) + b
		
		self.xt1 = xt1
		self.xt2 = xt2
		self.yt1 = yt1
		self.yt2 = yt2

		print(f"{xt1 = } {yt1 = }")
		print(f"{xt2 = } {yt2 = }")

		s1 = ((b - yt1) * (yp - yt1)) / ((xt1 - a) * (xt1 - xp) + e)
		s2 = ((b - yt2) * (yp - yt2)) / ((xt1 - a) * (xt1 - xp) + e)

		print(f"{s1 = } {s2 = }")


	def tangents_points_r1(self):
		e = 1e-20
		d = self.c1.y
		c = self.c1.x
		r1 = self.c1.r
		xp = self.xp
		yp = self.yp

		l_up = r1**2 * (xp - c)
		r_up = r1 * (yp - d) * ((xp - c)**2 + (yp - d)**2 - r1**2)**(1/2)
		down = (xp - c)**2 + (yp - d)**2

		print(f"xt = {r1=}**2 * ({xp=} - {c=})")
		print(f"xt = {r1=} * ({yp=} - {d=}) * (({xp=} - {c=})**2 + ({yp=} - {d=})**2 - {r1=}**2)**(1/2)")
		print(f"xt = ({xp=} - {c=})**2 + ({yp=} - {d=})**2")

		xt3 = ((l_up + r_up) / down) + c
		xt4 = ((l_up - r_up) / down) + c

		l_up = r1**2 * (yp - d)
		r_up = r1 * (xp - c) * ((xp - c)**2 + (yp - d)**2 - r1**2)**(1/2)
		down = (xp - c)**2 + (yp - d)**2

		yt3 = ((l_up - r_up) / down) + d
		yt4 = ((l_up + r_up) / down) + d

		print(f"{xt3 = } {yt3 = }")
		print(f"{xt4 = } {yt4 = }")

		s3 = ((d - yt3) * (yp * yt3)) / ((xt3 - c) * (xt3 - xp) + e)
		s4 = ((d - yt4) * (yp * yt4)) / ((xt3 - c) * (xt3 - xp) + e)

		print(f"{s3 = } {s4 = }")

		self.xt3 = xt3
		self.xt4 = xt4
		self.yt3 = yt3
		self.yt4 = yt4


	def valid(self):
		self.circle_dist()
		r0 = abs(self.c0.r)
		r1 = abs(self.c1.r)
		if self.D >= abs(r0 - r1):
			return True
		return False

	def do(self):
		self.tangent_lines_intersection()
		self.tangents_points_r0()
		self.tangents_points_r1()

		line_1 = (self.xt1, self.yt1), (self.xt3, self.yt3)
		line_2 = (self.xt2, self.yt2), (self.xt4, self.yt4)

		self.l1 = line_1
		self.l2 = line_2

	def line_1(self, x):
		yt1 = self.yt1
		xt1 = self.xt1
		xp = self.xp
		yp = self.yp

		y = ((x - xt1)/(xp - xt1))*(yp - yt1)+yt1
		return y

	def line_2(self, x):
		yt3 = self.yt3
		xt3 = self.xt3
		xp = self.xp
		yp = self.yp

		y = ((x - xt3)/(xp - xt3))*(yp - yt3)+yt3
		return y

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
		plt.legend(loc="upper left")
		plt.show()

	def get_tangents(self, start, ax):
		a = self.p[start]
		for i in range(start + 1, len(self.p)):
			b = self.p[i]
			t = Tangent(a, b)
			if t.valid():
				t.do()
				self.plot_tangents(ax, start, t)
				return
		
	def plot_tangents(self, ax, c, t):
		NUM_COLORS = 30
		cm = plt.get_cmap('gist_rainbow')
		# ax.plot(*t.l1, color=cm(1.*(c + 1)/NUM_COLORS), label="tan1")
		# ax.plot(*t.l2, color=cm(1.*(c + 1)/NUM_COLORS), label="tan2")
		# ax.scatter(t.xp, t.yp, color=cm(1.*(c + 2)/NUM_COLORS), s=10, label="P")
		ax.scatter(t.a, t.b, color=cm(1.*(c)/NUM_COLORS), s=10, label="o0")
		# ax.scatter(t.c, t.d, color=cm(1.*(c + 4)/NUM_COLORS), s=10, label="o1")
		min_x = min([t.c0.x, t.c1.x])
		max_x = max([t.c0.x, t.c1.x])
		xs = np.linspace(min_x, max_x, 10)
		ccc = 'k' if c % 2 else 'w'
		ax.plot(xs, t.line_1(xs), color=ccc, label="line1")
		ax.plot(xs, t.line_1(xs)+1, color='k', label="line1")
		ax.plot(xs, t.line_1(xs)-1, color='k', label="line1")
		ax.plot(xs, t.line_2(xs), color=ccc, label="line2")


	def plot_circles(self):
		print(len(self.p))
		NUM_COLORS = 30
		cm = plt.get_cmap('gist_rainbow')
		fig, ax = plt.subplots()

		circles = []
		for i, p in enumerate(self.p):
			c = plt.Circle((p.x, p.y), p.r, color=cm(1.*i/NUM_COLORS), fill=False, label=f"c{i}")
			circles.append(c)

			self.get_tangents(i, ax)


		for c in circles:
			ax.add_patch(c)
	
		ax.set_xlim((-10, 100))
		ax.set_ylim((-10, 100))

		# plt.legend(loc="upper left")
		plt.show()

def load(data_path):
	with open(data_path) as f:
		content = f.read()
		jsn = json.loads(content)
	l = Line()
	for e in jsn:
		x = e["pos"][2]
		y = e["pos"][1]
		z = e["pos"][0]
		r = e["cte"]
		l.add(Point(x, y, z, r))
	return l


if __name__ == "__main__":
	f = "one.json"
	f = "two.json"
	path = f"srcs/Scorer/datas/{f}"

	data = load(path)
	
	# data = Line()
	# data.add(Point(0, 0, 0, 1))
	# data.add(Point(3, -1, 0, 2))
	# data.add(Point(-4, -2, 0, 3))
	# data.add(Point(3, 4, 0, 1))

	data.plot_circles()
