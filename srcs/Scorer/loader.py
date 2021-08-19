import json
import Line, Point

def load(data_path):
	with open(data_path) as f:
		content = f.read()
		jsn = json.loads(content)

	l = Line()
	for e in jsn:
		l.add(Point(e))

	return l
