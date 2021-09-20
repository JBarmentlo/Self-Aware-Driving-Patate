from loader import load
from Circle import Circle
from RoadDeducer import RoadDeducer


if __name__ == "__main__":
	f = "one.json"
	f = "two.json"
	path = f"srcs/Scorer/test_datas/{f}"

	data = RoadDeducer()
	data.add(Circle(1, 1, 0, -3))
	data.add(Circle(2, 3, 0, -1))
	data.add(Circle(3, 5, 0, 1.1))
	data.add(Circle(5, 3, 0, -1))
	data.add(Circle(9, 5, 0, -3.5))
	data.add(Circle(3, 9, 0, 2))
	data.add(Circle(1, 12, 0, 1))
	data.add(Circle(-1, 15, 0, -1.1))

	# data = load(path)

	data.plot_circles()
