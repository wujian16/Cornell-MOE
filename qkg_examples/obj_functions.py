import numpy

class BraninNoNoise(object):
    def __init__(self):
        self._dim = 2
        self._search_domain = numpy.array([[0, 15],[-5,15]])
        self._hyper_domain = numpy.array([[1e-6, 1e6], [1e-6, 1e6], [1e-6, 1e6],
                                          [1e-6, 1e6], [1e-6, 1e6], [1e-6, 1e6]])
        self._num_init_pts = 3
        self._sample_var = 0.0
        self._min_value = 0.397887

    def evaluate_true(self, x):
        """ This function is usually evaluated on the square x_1 \in [-5, 10], x_2 \in [0, 15]. Global minimum
        is at x = [-pi, 12.275], [pi, 2.275] and [9.42478, 2.475] with minima f(x*) = 0.397887.

            :param x[2]: 2-dim numpy array
        """
        a = 1
        b = 5.1 / (4 * pow(numpy.pi, 2.0))
        c = 5 / numpy.pi
        r = 6
        s = 10
        t = 1 / (8 * numpy.pi)
        return numpy.array([(a * pow(x[1] - b * pow(x[0], 2.0) + c * x[0] - r, 2.0) + s * (1 - t) * numpy.cos(x[0]) + s),
                (2*a*(x[1] - b * pow(x[0], 2.0) + c * x[0] - r) * (-2* b * x[0] + c) + s * (1 - t) * (-numpy.sin(x[0]))),
                (2*a*(x[1] - b * pow(x[0], 2.0) + c * x[0] - r))])

    def evaluate(self, x):
        return self.evaluate_true(x)

class RosenbrockNoNoise(object):
    def __init__(self):

        self._dim = 3
        self._search_domain = numpy.repeat([[-2., 2.]], 3, axis=0)

        self._hyper_domain = numpy.array([[10., 1.0e8], [0.5, 10.], [0.5, 10.], [0.5, 10.], [0.01, 0.5],
                                          [0.01, 0.05], [0.01, 0.05], [0.01, 0.05]])
        self._num_init_pts = 3
        self._sample_var = 0.25
        self._min_value = 0.0


    def evaluate_true(self, x):
        """ Global minimum is 0 at (1, 1)

            :param x[2]: 2-dimension numpy array
        """
        value = 0.0
        for i in range(self._dim-1):
            value += pow(1. - x[i], 2.0) + 100. * pow(x[i+1] - pow(x[i], 2.0), 2.0)
        results = [value]
        for i in range(self._dim-1):
            results += [2.*(x[i]-1) - 400.*x[i]*(x[i+1]-pow(x[i], 2.0))]
        results += [200. * (x[self._dim-1]-pow(x[self._dim-2], 2.0))]
        return numpy.array(results)

    def evaluate(self, x):
        return self.evaluate_true(x)

class HartmannNoNoise(object):
    def __init__(self):
        self._dim = 6
        self._search_domain = numpy.repeat([[0., 1.]], self._dim, axis=0)
        self._hyper_domain = numpy.array([[0.1, 100.], [0.1, 10.], [0.1, 10.], [0.1, 10.],
                                          [0.1, 10.], [0.1, 10.], [0.1, 10.], [0.01, 0.5],
                                          [0.01, 0.5], [0.01, 0.5], [0.01, 0.5],
                                          [0.01, 0.5], [0.01, 0.5], [0.01, 0.5]])
        self._num_init_pts = 3
        self._sample_var = 0.25
        self._min_value = -3.32237
        self._num_observations = 6

    def evaluate_true(self, x):
        """ domain is x_i \in (0, 1) for i = 1, ..., 6
            Global minimum is -3.32237 at (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)

            :param x[6]: 6-dimension numpy array with domain stated above
        """
        alpha = numpy.array([1.0, 1.2, 3.0, 3.2])
        A = numpy.array([[10, 3, 17, 3.50, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8],
                         [17, 8, 0.05, 10, 0.1, 14]])
        P = 1.0e-4 * numpy.array([[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991],
                                  [2348, 1451, 3522, 2883, 3047, 6650], [4047, 8828, 8732, 5743, 1091, 381]])

        results = [0.0]*7

        for i in range(4):
            inner_value = 0.0
            for j in range(self._dim):
                inner_value -= A[i, j] * pow(x[j] - P[i, j], 2.0)
            results[0] -= alpha[i] * numpy.exp(inner_value)
            for j in range(self._dim):
                results[j+1] -= (alpha[i] * numpy.exp(inner_value)) * ((-2) * A[i,j] * (x[j] - P[i, j]))

        return numpy.array(results)

    def evaluate(self, x):
        return self.evaluate_true(x)