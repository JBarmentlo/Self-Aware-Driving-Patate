from Circle import Circle


class TangentsExtern():
	# http://www.ambrsoft.com/TrigoCalc/Circles2/Circles2Tangent_.htm

	def __init__(self, ca: Circle, cb: Circle) -> None:
		if ca.r > cb.r:
			self.c0 = ca
			self.c1 = cb
		else:
			self.c1 = ca
			self.c0 = cb
		self.r0 = abs(self.c0.r)
		self.r1 = abs(self.c1.r)

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
		r0 = self.r0
		d = self.c1.y
		c = self.c1.x
		r1 = self.r1

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
		r0 = self.r0
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
		r1 = self.r1
		xp = self.xp
		yp = self.yp

		l_up = r1**2 * (xp - c)
		r_up = r1 * (yp - d) * ((xp - c)**2 + (yp - d)**2 - r1**2)**(1/2)
		down = (xp - c)**2 + (yp - d)**2

		print(f"xt = {r1=}**2 * ({xp=} - {c=})")
		print(
			f"xt = {r1=} * ({yp=} - {d=}) * (({xp=} - {c=})**2 + ({yp=} - {d=})**2 - {r1=}**2)**(1/2)")
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
		r0 = self.r0
		r1 = self.r1
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

