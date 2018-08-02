""" Default and example column probability density function classes. """

import numpy as np


class Distribution(object):
    """ A base class for the distributions below and those defined by users. """

    def sample(self):
        """ Raise a NotImplementedError by default. """

        raise NotImplementedError("You must define a sample method.")


class Gamma(Distribution):
    """ Continuous column pdf given by the gamma distribution. """

    alpha_limits = [0, 10]
    theta_limits = [0, 10]

    def __init__(self):

        self.alpha = np.random.uniform(*self.alpha_limits)
        self.theta = np.random.uniform(*self.theta_limits)

    def __repr__(self):

        return f"Gamma(alpha={self.alpha:.2f}, theta={self.theta:.2f})"

    def sample(self, nrows):
        """ Take a sample of size `nrows` from the gamma distribution with
        shape and scale parameters given by `alpha` and `theta` respectively.
        """

        return np.random.gamma(shape=self.alpha, scale=self.theta, size=nrows)


class Normal(Distribution):
    """ Continuous column pdf given by the normal distribution. """

    mean_limits = [-10, 10]
    std_limits = [0, 10]

    def __init__(self):

        self.mean = np.random.uniform(*self.mean_limits)
        self.std = np.random.uniform(*self.std_limits)

    def __repr__(self):

        return f"Normal(mean={self.mean:.2f}, std={self.std:.2f})"

    def sample(self, nrows):
        """ Take a sample of size `nrows` from the normal distribution with
        mean and standard deviation given by `mean` and `std` respectively. """

        return np.random.normal(loc=self.mean, scale=self.std, size=nrows)


class Poisson(Distribution):
    """ Discrete column pdf given by the Poisson distribution. """

    lam_limits = [0, 10]

    def __init__(self):

        self.lam = np.random.uniform(*self.lam_limits)

    def __repr__(self):

        return f"Poisson(lambda={self.lam:.2f})"

    def sample(self, nrows):
        """ Take a sample of size `nrows` from the Poisson distribution with
        parameter `lam`. Seeded for reproducibility. """

        return np.random.poisson(lam=self.lam, size=nrows)
