""" All currently implemented, discrete distribution classes. """

import numpy as np

from .base import Distribution


class Bernoulli(Distribution):
    """ Discrete column class given by the Bernoulli distribution. That is, a
    binomial distribution with exactly one trial.

    Attributes
    ----------
    name : str
        Name of the distribution, :code:`"Bernoulli"`.
    dtype : str
        Preferred datatype of distribution, :code:`"int"`.
    param_limits : dict
        A dictionary of the limits on the distribution parameter. Defaults to
        :code:`[0, 1]` for :code:`prob`.
    prob : float
        The success probability, sampled from :code:`param_limits["prob"]`.
        `Instance attribute.`
    """

    name = "Bernoulli"
    dtype = "int"
    hard_limits = {"prob": [0, 1]}
    param_limits = {"prob": [0, 1]}

    def __init__(self):

        self.prob = np.random.uniform(*self.param_limits["prob"])

    def sample(self, nrows):
        """ Take a sample of size :code:`nrows` from the Bernoulli distribution
        with success probability :code:`prob`. """

        return np.random.binomial(n=1, p=self.prob, size=nrows)


class Poisson(Distribution):
    """ Discrete column class given by the Poisson distribution.

    Attributes
    ----------
    name : str
        Name of distribution, :code:`"Poisson"`.
    dtype : str
        Preferred datatype of distribution, :code:`"int"`.
    param_limits : dict
        A dictionary of the limits of the distribution parameter. Defaults to
        :code:`[0, 10]` for :code:`lam`.
    lam : float
        The rate parameter, sampled from :code:`param_limits["lam"]`. `Instance
        attribute.`
    """

    name = "Poisson"
    dtype = "int"
    hard_limits = {"lam": [0, 10]}
    param_limits = {"lam": [0, 10]}

    def __init__(self):

        self.lam = np.random.uniform(*self.param_limits["lam"])

    def sample(self, nrows):
        """ Take a sample of size :code:`nrows` from the Poisson distribution
        with rate parameter :code:`lam`. """

        return np.random.poisson(lam=self.lam, size=nrows)
