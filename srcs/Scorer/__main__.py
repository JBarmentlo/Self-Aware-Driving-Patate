from loader import load
from Circle import Circle
from RoadDeducer import RoadDeducer


if __name__ == "__main__":
	f = "one.json"
	f = "two.json"
	path = f"srcs/Scorer/datas/{f}"

	data = RoadDeducer()
	# data.add(Circle(0, 0, 0, 1))
	# data.add(Circle(3, -1, 0, 2))
	data.add(Circle(5, 2, 0, 1))
	data.add(Circle(3, 4, 0, 1))


	# data = load(path)

	data.plot_circles()
