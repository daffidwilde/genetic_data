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
    prob_limits : list
        Limits on the success probability parameter, :code:`prob`. Defaults to
        :code:`[0, 1]`.
    prob : float
        The success probability, sampled from :code:`prob_limits`. `Instance
        attribute.`
    """

    name = "Bernoulli"
    dtype = "int"
    prob_limits = [0, 1]

    def __init__(self):

        self.prob = np.random.uniform(*self.prob_limits)

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
    lam_limits : list
        Limits on the rate parameter, :code:`lam`. Defaults to :code:`[0, 10]`.
    lam : float
        The rate parameter, sampled from :code:`lam_limits`. `Instance
        attribute.`
    """

    name = "Poisson"
    dtype = "int"
    lam_limits = [0, 10]

    def __init__(self):

        self.lam = np.random.uniform(*self.lam_limits)

    def sample(self, nrows):
        """ Take a sample of size :code:`nrows` from the Poisson distribution
        with rate parameter :code:`lam`. """

        return np.random.poisson(lam=self.lam, size=nrows)
