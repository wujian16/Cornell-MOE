# -*- coding: utf-8 -*-
"""Implementations of deep kernel functions for use with
:mod:`moe.optimal_learning.python.python_version.log_likelihood` and :mod:`moe.optimal_learning.python.python_version.scalable_gaussian_process`.

This file contains implementations of CovarianceInterface. Currently, we have the
Deep Kernel consists of a neural network projection and a gaussian process layer, supporting:

* covariance
* grad_covariance
* hyperparameter_grad_covariance

It also contains a few utilities for computing common mathematical quantities and
initialization.

Gradient (spatial and hyperparameter) functions are implemented by tf.gradients for fast performance.

"""
import numpy
import tensorflow as tf

from moe.optimal_learning.python.constant import DEEP_KERNEL_COVARIANCE_TYPE
#from moe.optimal_learning.python.interfaces.covariance_interface import CovarianceInterface

class DeepKernel(object):

    r"""Implement the deep kernel function.

    The function:
    ``cov(x_1, x_2) = \alpha * \exp(-1/2 * ((g(x_1; theta) - g(x_2; theta))^T * L * (g(x_1; theta) - g(x_2; theta))) )``
    where L is the diagonal matrix with i-th diagonal entry ``1/lengths[i]/lengths[i]``

    This covariance object has paramters ``theta`` + ``dim+1`` hyperparameters: ``\alpha, lengths_i``
    """

    covariance_type = DEEP_KERNEL_COVARIANCE_TYPE

    def __init__(self, hyperparameters):
        r"""Construct a deep kernel object with the specified hyperparameters.

        :param hyperparameters: hyperparameters of the deep kernel function;
        a list of numpy.ndarray
        :type hyperparameters: list of numpy.ndarray
        """
        self._hyperparameters = hyperparameters
        self._W_0, \
        self._W_1, \
        self._W_2, \
        self._W_3, \
        self._b_0, \
        self._b_1, \
        self._b_2, \
        self._b_3, \
        self._sigma, \
        self._lengths_scale = self._hyperparameters

    @property
    def num_hyperparameters(self):
        """Return the length of hyperparameters of this covariance function."""
        return len(self._hyperparameters)

    def get_hyperparameters(self):
        """Get the hyperparameters (array of float64 with shape (num_hyperparameters)) of this covariance."""
        return [numpy.copy(param.eval()) for param in self._hyperparameters]

    def set_hyperparameters(self, hyperparameters):
        """Set hyperparameters to the specified hyperparameters; ordering must match.
        list of tf.ndarray"""
        self._hyperparameters = [tf.constant(param) for param in hyperparameters]
        self._lengths_sq = numpy.copy(hyperparameters[-1])
        self._lengths_sq *= self._lengths_sq

    hyperparameters = property(get_hyperparameters, set_hyperparameters)

    def neural_network(self, X):
        """define the neural network part
        Parameters
        ----------
        X: numpy.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        """
        h = tf.tanh(tf.matmul(X, self._W_0) + self._b_0)
        h = tf.tanh(tf.matmul(h, self._W_1) + self._b_1)
        h = tf.tanh(tf.matmul(h, self._W_2) + self._b_2)
        h = tf.tanh(tf.matmul(h, self._W_3) + self._b_3)
        return h

    def square_dist(self, points_one, points_two = None):
        r"""Compute the square distances of two sets of points, cov(``points_one``, ``points_two``).

        Square Exponential: ``dis(x_1, x_2) = ((x_1 - x_2)^T * L * (x_1 - x_2)) ``

        :param points_one: first input, the point ``x``
        :type point_one: array of float64 with shape (N1, dim)
        :param points_two: second input, the point ``y``
        :type point_two: array of float64 with shape (N2, dim)
        :return: the square distance matrix (tensor) between the input points
        :rtype: tensor of float64 with shape (N1, N2)
        """
        points_one = tf.divide(points_one, self._lengths_scale)
        set_sum1 = tf.reduce_sum(tf.square(points_one), 1)
        if points_two is None:
            return -2 * tf.matmul(points_one, tf.transpose(points_one)) + \
                   tf.reshape(set_sum1, (-1, 1)) + tf.reshape(set_sum1, (1, -1))
        else:
            points_two = tf.divide(points_two, self._lengths_scale)
            set_sum2 = tf.reduce_sum(tf.square(points_two), 1)
            return -2 * tf.matmul(points_one, tf.transpose(points_two)) + \
                   tf.reshape(set_sum1, (-1, 1)) + tf.reshape(set_sum2, (1, -1))

    def covariance(self, points_one, points_two=None):
        r"""Compute the square exponential covariance function of two points, cov(``point_one``, ``point_two``).

        Square Exponential: ``cov(x_1, x_2) = \alpha * \exp(-1/2 * ((x_1 - x_2)^T * L * (x_1 - x_2)) )``

        .. Note:: comments are copied from the matching method comments of
          :class:`moe.optimal_learning.python.interfaces.covariance_interface.CovarianceInterface`.

        The covariance function is guaranteed to be symmetric by definition: ``covariance(x, y) = covariance(y, x)``.
        This function is also positive definite by definition.

        :param points_one: first input, the point ``x``
        :type point_one: array of float64 with shape (N1, dim)
        :param points_two: second input, the point ``y``
        :type point_two: array of float64 with shape (N2, dim)
        :return: the covariance tensor between the input points
        :rtype: tensor of float64 with shape (N1, N2)

        """
        return (self._sigma ** 2) * tf.exp(-0.5 * self.square_dist(points_one, points_two))

    def grad_covariance(self, point_one, point_two):
        r"""Compute the gradient of self.covariance(point_one, point_two) with respect to the FIRST argument, point_one.

        Gradient of Square Exponential (wrt ``x_1``):
        ``\pderiv{cov(x_1, x_2)}{x_{1,i}} = (x_{2,i} - x_{1,i}) / L_{i}^2 * cov(x_1, x_2)``

        .. Note:: comments are copied from the matching method comments of
          :class:`moe.optimal_learning.python.interfaces.covariance_interface.CovarianceInterface`.

        This distinction is important for maintaining the desired symmetry.  ``Cov(x, y) = Cov(y, x)``.
        Additionally, ``\pderiv{Cov(x, y)}{x} = \pderiv{Cov(y, x)}{x}``.
        However, in general, ``\pderiv{Cov(x, y)}{x} != \pderiv{Cov(y, x)}{y}`` (NOT equal!  These may differ by a negative sign)

        Hence to avoid separate implementations for differentiating against first vs second argument, this function only handles
        differentiation against the first argument.  If you need ``\pderiv{Cov(y, x)}{x}``, just swap points x and y.

        :param point_one: first input, the point ``x``
        :type point_one: array of float64 with shape (dim)
        :param point_two: second input, the point ``y``
        :type point_two: array of float64 with shape (dim)
        :return: grad_cov: i-th entry is ``\pderiv{cov(x_1, x_2)}{x_i}``
        :rtype: array of float64 with shape (dim)

        """
        point_one = tf.constant(point_one)
        point_one = tf.cast(point_one, tf.float32)
        point_two = tf.cast(point_two, tf.float32)
        grad_cov = tf.gradients(self.covariance(tf.reshape(point_one, (1, -1)), tf.reshape(point_two, (1, -1))), point_one)[0]
        return grad_cov