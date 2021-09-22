import math
import numpy

class Point:
    '''
    A point on the track with its distance to the center.
    '''

    def __init__(self, pos, dist):
        self.pos = pos
        self.dist = dist

def compute_distance(p0, p1):
    '''
    Computes the distance between `p0` and `p1`, projected onto the center of the track.
    '''
    delta_pos = p1.pos - p0.pos
    delta_dist = p1.dist - p0.dist

    delta_pos_norm = numpy.linalg.norm(delta_pos)
    sqr_dist = (delta_pos_norm * delta_pos_norm) - (delta_dist * delta_dist)
    if sqr_dist >= 0.00001:
        return math.sqrt(sqr_dist)
    else:
        return 0.

def projection_angle(p0, p1, proj_dist):
    '''
    Computes the angle of the projected vector relative to the center of the track.
    '''
    delta_pos = p1.pos - p0.pos
    delta_pos_len = numpy.linalg.norm(delta_pos)
    if delta_pos_len >= 0.00001:
        return math.acos(numpy.linalg.norm(proj_dist) / delta_pos_len)
    else:
        return 0.

def get_projected_vector(p0, p1):
    '''
    Computes the delta vector projected onto the center of the track.
    '''
    delta_pos = p1.pos - p0.pos
    proj_dist = compute_distance(p0, p1)
    delta_angle = projection_angle(p0, p1, proj_dist)

    if p1.dist < 0.:
        delta_angle = -delta_angle

    proj = numpy.array([
        delta_pos[0] * math.cos(delta_angle) - delta_pos[1] * math.sin(delta_angle),
        0.,
        delta_pos[0] * math.sin(delta_angle) + delta_pos[1] * math.cos(delta_angle),
    ])
    proj /= numpy.linalg.norm(proj)
    proj *= proj_dist
    return proj

class DistanceTracker():
    def __init__(self, start_pos, start_cte):
        '''
        `start_pos` is the starting position of the car.
        `start_cte` is the starting Cross Track Error of the car.
        '''
        self.prev = Point(start_pos, start_cte)
        self.prev_proj = numpy.array([0., 0., 0.])
        self.total_distance = 0.

    def get_total_distance(self):
        '''
        Returns the total distance travelled since the beginning.
        This function can be used to compute the score.
        '''
        return self.total_distance

    def next(self, pos, cte):
        '''
        Updates the distance using the given new position and cte.
        This function must be called at each frames of the simulator to update the position and the travelled distance.

        `start_pos` is the starting position of the car.
        `start_cte` is the starting Cross Track Error of the car.
        '''
        pos = numpy.array(pos)
        curr = Point(pos, cte)
        proj = get_projected_vector(curr, self.prev)

        proj_len = numpy.linalg.norm(proj)
        print(f"{self.prev_proj = }, {proj = }")
        direction = numpy.dot(self.prev_proj, proj)
        if direction >= -0.001:
            self.total_distance += proj_len
        elif direction:
            self.total_distance -= proj_len

        self.prev = curr
        self.prev_proj = proj

if __name__ == "__main__":
    d = DistanceTracker(numpy.array([0., 0.]), 0.)
    d.next(numpy.array([1., 1.]), 1.)
    d.next(numpy.array([-3., 2.]), 1.)

    # If uncommented, must partially cancel previous movements
    d.next(numpy.array([0., 0.]), 0.)

    print(d.get_total_distance())
