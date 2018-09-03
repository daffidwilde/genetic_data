""" All currently implemented, continuous distributions. """

import numpy as np

from .base import Distribution

class Gamma(Distribution):
    """ Continuous column class given by the gamma distribution.

    Attributes
    ----------
    name : str
        Name of the distribution, :code:`"Gamma"`.
    dtype : str
        Preferred datatype of distribution, :code:`"float"`.
    alpha_limits : list
        Limits on the shape parameter, :code:`alpha`. Defaults to
        :code:`[0, 10]`.
    theta_limits : list
        Limits on the scale parameter, :code:`theta`. Defaults to
        :code:`[0, 10]`.
    alpha : float
        The shape parameter sampled from :code:`alpha_limits`. `Instance
        attribute.`
    theta : float
        The scale parameter sampled from :code:`theta_limits`. `Instance
        attribute.`
    """

    name = "Gamma"
    dtype = "float"
    alpha_limits = [0, 10]
    theta_limits = [0, 10]

    def __init__(self):

        self.alpha = np.random.uniform(*self.alpha_limits)
        self.theta = np.random.uniform(*self.theta_limits)

    def sample(self, nrows):
        """ Take a sample of size :code:`nrows` from the gamma distribution with
        shape and scale parameters given by :code:`alpha` and :code:`theta`
        respectively. """

        return np.random.gamma(shape=self.alpha, scale=self.theta, size=nrows)


class Normal(Distribution):
    """ Continuous column class given by the normal distribution.

    Attributes
    ----------
    name : str
        Name of the distribution, :code:`"Normal"`.
    dtype : str
        Preferred datatype of distribution, :code:`"float"`.
    mean_limits : list
        Limits on the mean, :code:`mean`. Defaults to :code:`[-10, 10]`.
    std_limits : list
        Limits on the standard deviation, :code:`std`. Defaults to
        :code:`[0, 10]`.
    mean : float
        The mean, sampled from :code:`mean_limits`. `Instance attribute.`
    std : float
        The standard deviation, sampled from :code:`std_limits`. `Instance
        attribute.`
    """

    name = "Normal"
    dtype = "float"
    mean_limits = [-10, 10]
    std_limits = [0, 10]

    def __init__(self):

        self.mean = np.random.uniform(*self.mean_limits)
        self.std = np.random.uniform(*self.std_limits)

    def sample(self, nrows):
        """ Take a sample of size :code:`nrows` from the normal distribution
        with mean and standard deviation given by :code:`mean` and :code:`std`
        respectively. """

        return np.random.normal(loc=self.mean, scale=self.std, size=nrows)
