import math
import numpy
import scipy
from scipy import optimize
from scipy.optimize import minimize
from scipy.stats import norm

class Probe:
    def __init__(self, dim):
        self.dim = dim

    def probe(self, x):
        pass

def gaussian_kernel(alpha, x0, x1):
    distance = numpy.linalg.norm(x0 - x1)
    return alpha * math.exp(-(distance * distance))

def data_y_to_vec(data):
    v = numpy.ndarray(shape=(len(data), 1))
    for i in range(len(data)):
        v[i][0] = data[i][1]
    return v

def compute_mean(data, x):
    a = numpy.ndarray(shape=(1, len(data)))
    for i in range(len(data)):
        a[0][i] = gaussian_kernel(1., data[i][0], x)

    b = numpy.ndarray(shape=(len(data), len(data)))
    for i in range(len(data)):
        for j in range(i, len(data)):
            val = gaussian_kernel(1., data[i][0], data[j][0])
            b[i][j] = val
            b[j][i] = val
    b = numpy.linalg.inv(b)

    c = data_y_to_vec(data)
    return numpy.matmul(a, numpy.matmul(b, c))[0][0]

def compute_std_deviation(data, x):
    a = gaussian_kernel(1., x, x)

    b = numpy.ndarray(shape=(1, len(data)))
    for i in range(len(data)):
        b[0][i] = gaussian_kernel(1., x, data[i][0])

    c = numpy.ndarray(shape=(len(data), len(data)))
    for i in range(len(data)):
        for j in range(i, len(data)):
            val = gaussian_kernel(1., data[i][0], data[j][0])
            c[i][j] = val
            c[j][i] = val
    c = numpy.linalg.inv(c)

    d = numpy.ndarray(shape=(len(data), 1))
    for i in range(len(data)):
        d[i][0] = gaussian_kernel(1., data[i][0], x)

    # TODO Keep abs?
    return math.sqrt(abs(a - (numpy.matmul(b, numpy.matmul(c, d)))))

def negative_expected_improvement(x, samples, max_sample):
    mean = compute_mean(samples, x)
    std_deviation = compute_std_deviation(samples, x)

    delta = mean - max_sample

    A = 0
    if abs(std_deviation) > 0.0001:
        A = delta / std_deviation

    density = norm.pdf(x=A, loc=0., scale=1.)
    cumulative_density = norm.cdf(x=A, loc=0., scale=1.)

    return -(max(delta, 0.) + std_deviation * density - abs(delta) * cumulative_density)

def bayesian_optimization(probe, random_iterations, max_iterations):
    n = 0
    samples = []
    max_index = 0

    while n < random_iterations:
        x = numpy.random.uniform(0., 1., size=(probe.dim))

        y = probe.probe(x)
        samples.append((x, y))
        if y > samples[max_index][1]:
            max_index = len(samples) - 1

        n += 1

    while n < max_iterations:
        begin = numpy.random.uniform(0., 1., size=(probe.dim))
        max_sample = samples[max_index][1]
        print('max: ', max_sample)
        x = scipy.optimize.minimize(negative_expected_improvement, begin, args=(samples, max_sample), bounds=scipy.optimize.Bounds(0., 1.), method='L-BFGS-B').x

        y = probe.probe(x)
        samples.append((x, y))
        if y > max_sample:
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
        print('x:', x)
        print('y:', 2. * math.sin(x[1] * x[1]) + math.cos(2. * x[0]))
        return 2. * math.sin(x[1] * x[1]) + math.cos(2. * x[0])

if __name__ == "__main__":
    probe = TestProbe()
    result = bayesian_optimization(probe, 25, 100)
    print('result:', result)
