from typing import Deque

def is_stuck(speeds : Deque, limit):
	if (len(speeds) != speeds.maxlen):
		return False

	sum = 0
	for s in speeds:
		sum += s
	
	avg = sum / len(speeds)
	if (avg < limit):
		print(f"\nSTUCK {avg}\n")
	return (avg < limit)
