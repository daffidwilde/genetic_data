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
    param_limits : dict
        A dictionary of limits on the distribution parameters. Defaults to
        :code:`[0, 10]` for both :code:`alpha` and :code:`theta`.
    alpha : float
        The shape parameter sampled from :code:`param_limits["alpha"]`.
        `Instance attribute.`
    theta : float
        The scale parameter sampled from :code:`param_limits["theta"]`.
        `Instance attribute.`
    """

    name = "Gamma"
    dtype = "float"
    hard_limits = {"alpha": [0, 10], "theta": [0, 10]}
    param_limits = {"alpha": [0, 10], "theta": [0, 10]}

    def __init__(self):

        self.alpha = np.random.uniform(*self.param_limits["alpha"])
        self.theta = np.random.uniform(*self.param_limits["theta"])

        super().__init__()

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
    param_limits : dict
        A dictionary of limits on the distribution parameters. Defaults to
        :code:`[-10, 10]` for :code:`mean` and :code:`[0, 10]` for :code:`std`.
    mean : float
        The mean, sampled from :code:`param_limits["mean"]`. `Instance
        attribute.`
    std : float
        The standard deviation, sampled from :code:`param_limits["std"]`.
        `Instance attribute.`
    """

    name = "Normal"
    dtype = "float"
    hard_limits = {"mean": [-10, 10], "std": [0, 10]}
    param_limits = {"mean": [-10, 10], "std": [0, 10]}

    def __init__(self):

        self.mean = np.random.uniform(*self.param_limits["mean"])
        self.std = np.random.uniform(*self.param_limits["std"])

        super().__init__()

    def sample(self, nrows):
        """ Take a sample of size :code:`nrows` from the normal distribution
        with mean and standard deviation given by :code:`mean` and :code:`std`
        respectively. """

        return np.random.normal(loc=self.mean, scale=self.std, size=nrows)
