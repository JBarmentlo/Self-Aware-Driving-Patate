import json
from Circle import Circle
from RoadDeducer import RoadDeducer

def load(data_path):
	with open(data_path) as f:
		content = f.read()
		jsn = json.loads(content)
	l = RoadDeducer()
	for e in jsn:
		x = e["pos"][2]
		y = e["pos"][1]
		z = e["pos"][0]
		r = e["cte"]
		l.add(Circle(x, y, z, r))
	return l
