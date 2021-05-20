import math
import numpy
import scipy
from scipy import optimize
from scipy.optimize import minimize

class Probe:
    def __init__(self, dim):
        self.dim = dim

    def probe(self, x):
        pass

def gaussian_kernel(alpha, x0, x1):
    distance = numpy.linalg.norm(x0 - x1)
    return alpha * math.exp(-(distance * distance))

def data_y_to_vec(data):
    v = numpy.ndarray(shape=(data.len(), 1))
    for i in range(data.len()):
        v[i] = data[i]
    return v

def compute_mean(data, x):
    a = numpy.ndarray(shape=(1, len(data)))
    for i in range(len(data)):
        a[0][i] = gaussian_kernel(1., data[i][0], x)

    b = numpy.ndarray(shape=(len(data), len(data)))
    for i in range(len(data)):
        for j in range(len(data)):
            b[i][j] = 1.
    print('= ', b)
    b = numpy.linalg.inv(b)

    c = data_y_to_vec(data)
    return a * (numpy.matmul(b * c))

def compute_std_deviation(data, x):
    a = gaussian_kernel(1., x, x)

    b = numpy.ndarray(shape=(1, len(data)))
    for i in range(len(data)):
        b[0][i] = gaussian_kernel(1., data[i][0], x)

    c = numpy.ndarray(shape=(len(data), len(data)))
    for i in range(len(data)):
        for j in range(len(data)):
            c[i][j] = 1.
    c = numpy.linalg.inv(b)

    d = numpy.ndarray(shape=(len(data), 1))
    for i in range(len(data)):
        d[i][0] = gaussian_kernel(1., data[i][0], x)

    return sqrt(a - (b * (c * d)))

def negative_expected_improvement(x, samples, max_sample):
    mean = compute_mean(samples, x)
    std_deviation = compute_std_deviation(samples, x)

    delta = mean - max_sample

    A = delta / std_deviation

    density = scipy.stats.norm.pdf(x=A, loc=mean, scale=std_deviation)
    cumulative_density = scipy.stats.norm.cdf(x=A, loc=mean, scale=std_deviation)

    return -(maxf(delta, 0.) + std_deviation * density - abs(delta) * cumulative_density)

def bayesian_optimization(probe, random_iterations, max_iterations):
    n = 0
    samples = []
    max_index = 0

    while n < random_iterations:
        # TODO Take range as parameter?
        x = numpy.random.uniform(-100., 100., size=(probe.dim))

        y = probe.probe(x)
        samples.append((x, y))
        if y > samples[max_index][1]:
            max_index = len(samples) - 1

        n += 1

    while n < max_iterations:
        begin = 0 # TODO
        max_sample = samples[max_index][1]
        x = scipy.optimize.minimize(negative_expected_improvement, begin, args=(samples, max_sample), method='L-BFGS-B')

        y = probe.probe(x)
        samples.append((x, y))
        if y > samples[max_index][1]:
            max_index = len(samples) - 1

        n += 1

    # TODO Do another maximization on the final function?
    return samples[max_index][0]



# ------------------------------------------------------------------------------
#    Testing
# ------------------------------------------------------------------------------



class TestProbe(Probe):
    def __init__(self):
       super().__init__(2)

    def probe(self, x):
        return x[1] * x[1] - x[0] * x[0]

if __name__ == "__main__":
    probe = TestProbe()
    result = bayesian_optimization(probe, 25, 100)
    print('-> ', result)
