import math
import numpy

class Point:
    def __init__(self, pos, dist):
        self.pos = pos
        self.dist = dist

def compute_distance(p0, p1):
    delta_pos = p1.pos - p0.pos
    delta_dist = p1.dist - p0.dist

    delta_pos_norm = numpy.linalg.norm(delta_pos)
    sqr_dist = (delta_pos_norm * delta_pos_norm) - (delta_dist * delta_dist)
    if sqr_dist >= 0.00001:
        return math.sqrt(sqr_dist)
    else:
        return 0.

def projection_angle(p0, p1, proj_dist):
    delta_pos = p1.pos - p0.pos
    delta_pos_len = numpy.linalg.norm(delta_pos)
    if delta_pos_len >= 0.00001:
        return math.acos(numpy.linalg.norm(proj_dist) / delta_pos_len)
    else:
        return 0.

def get_projected_vector(p0, p1):
    delta_pos = p1.pos - p0.pos
    proj_dist = compute_distance(p0, p1)
    delta_angle = projection_angle(p0, p1, proj_dist)

    if p1.dist < 0.:
        delta_angle = -delta_angle

    return numpy.array([
        delta_pos[0] * math.cos(delta_angle) - delta_pos[1] * math.sin(delta_angle),
        delta_pos[0] * math.sin(delta_angle) + delta_pos[1] * math.cos(delta_angle),
    ])

if __name__ == "__main__":
    print(get_projected_vector(Point(numpy.array([0., 5.]), 1.), Point(numpy.array([3., 1.]), -1.)))
