import numpy
import scipy

class Probe:
	def __init__(self, dim):
		self.dim = dim

	def probe(self, x):
		pass

def gaussian_kernel(alpha, x0, x1):
	distance = (x0 - x1).length()
	return alpha * exp(-(distance * distance))

def expected_improvement(x, max_sample):
	mean = 0 # TODO
	std_deviation = 0 # TODO

	delta = mean - max_sample

	A = delta / std_deviation

	density = scipy.stats.norm.pdf(x=A, loc=mean, scale=std_deviation)
	cumulative_density = scipy.stats.norm.cdf(x=A, loc=mean, scale=std_deviation)

	return maxf(delta, 0.) + std_deviation * density - abs(delta) * cumulative_density

def bayesian_optimization(probe, random_iterations, max_iterations):
	n = 0
	samples = []
	max_index = 0

	while n < random_iterations:
		x = numpy.random.uniform(-100., 100., size=(probe.dim))

		y = probe.probe(x)
		samples.append((x, y))
		if y > samples[max_index]:
			max_index = samples.len() - 1

		n += 1

	while n < max_iterations:
		x = scipy.optimize.minimize(expected_improvement, args=(samples[max_index][1]), method='L-BFGS-B')

		y = probe.probe(x)
		samples.append((x, y))
		if y > samples[max_index]:
			max_index = samples.len() - 1

		n += 1

	return 0 # TODO
