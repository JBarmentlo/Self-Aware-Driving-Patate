from loader import load
from Circle import Circle
from RoadDeducer import RoadDeducer


if __name__ == "__main__":
	f = "one.json"
	f = "two.json"
	path = f"srcs/Scorer/datas/{f}"

	data = RoadDeducer()
	data.add(Circle(0, 0, 0, 1))
	data.add(Circle(3, -1, 0, 2))
	data.add(Circle(5, 2, 0, 1))
	data.add(Circle(2, 5, 0, -2))
	data.add(Circle(7, 9, 0, 5))
	data.add(Circle(20, 5, 0, 3))

	data.add(Circle(17, 15, 0, 1))
	data.add(Circle(23, 20, 0, -6))
	data.add(Circle(25, 35, 0, 4))
	data.add(Circle(35, 35, 0, 7))

	# data = load(path)

	data.plot_circles()
