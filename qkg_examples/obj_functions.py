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
        t = self.evaluate_true(x)
        return t