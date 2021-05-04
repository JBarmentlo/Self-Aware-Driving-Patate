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

def data_y_to_vec(data):
	v = numpy.ndarray(size=(data.len(), 1))
	for i in range(data.len()):
		v[i] = data[i]
	return v

def compute_mean(data, x):
	a = numpy.ndarray(size=(1, data.len()))
	for i in range(data.len()):
		a[0][i] = gaussian_kernel(data[i], x)

	b = numpy.ndarray(size=(data.len(), data.len()))
	val = gaussian_kernel(data[i], data[i])
	for i in range(data.len()):
		for j in range(data.len()):
			b[i][j] = val
	b = numpy.linalg.inv(b)

	c = data_y_to_vec(data)
	return a * (numpy.matmul(b * c))

def compute_std_deviation(data, x):
	a = gaussian_kernel(x, x)

	b = numpy.ndarray(size=(1, data.len()))
	for i in range(data.len()):
		b[0][i] = gaussian_kernel(data[i], x)

	c = numpy.ndarray(size=(data.len(), data.len()))
	val = gaussian_kernel(data[i], data[i])
	for i in range(data.len()):
		for j in range(data.len()):
			c[i][j] = val
	c = numpy.linalg.inv(b)

	d = numpy.ndarray(size=(data.len(), 1))
	for i in range(data.len()):
		d[i][0] = gaussian_kernel(data[i], x)

	return sqrt(a - (b * (c * d)))

def expected_improvement(x, samples, max_sample):
	mean = compute_mean(data, x)
	std_deviation = compute_std_deviation(data, x)

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
		# TODO Take range as parameter?
		x = numpy.random.uniform(-100., 100., size=(probe.dim))

		y = probe.probe(x)
		samples.append((x, y))
		if y > samples[max_index]:
			max_index = samples.len() - 1

		n += 1

	while n < max_iterations:
		x = scipy.optimize.maximize(expected_improvement, args=(samples, samples[max_index][1]), method='L-BFGS-B')

		y = probe.probe(x)
		samples.append((x, y))
		if y > samples[max_index]:
			max_index = samples.len() - 1

		n += 1

	# TODO Do another maximization on the final function?
	return samples[max_index][0]
