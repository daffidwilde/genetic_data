""" Default and example column probability density function classes. """

import numpy as np


class Distribution(object):
    """ A base class for the distributions below and those defined by users. """

    name = "Distribution"

    def sample(self):
        """ Raise a NotImplementedError by default. """

        raise NotImplementedError("You must define a sample method.")

    def __repr__(self):

        params = ""
        for key, val in self.__dict__.items():
            params += f"{key}={val:.2f}, "
        params = params[:-2]

        return f"{self.name}({params})"

    def to_tuple(self):
        """ Returns the type of distribution, and the names and values of all
        parameters as a tuple. This is used for the saving of data and little
        else. """

        out = [self.name]
        for key, val in self.__dict__.items():
            out.append(key)
            out.append(val)

        return tuple(out)


# ========================
# CONTINUOUS DISTRIBUTIONS
# ========================


class Gamma(Distribution):
    """ Continuous column pdf given by the gamma distribution. """

    name = "Gamma"
    dtype = "float"
    alpha_limits = [0, 10]
    theta_limits = [0, 10]

    def __init__(self):

        self.alpha = np.random.uniform(*self.alpha_limits)
        self.theta = np.random.uniform(*self.theta_limits)

    def sample(self, nrows):
        """ Take a sample of size `nrows` from the gamma distribution with
        shape and scale parameters given by `alpha` and `theta` respectively.
        """

        return np.random.gamma(shape=self.alpha, scale=self.theta, size=nrows)


class Normal(Distribution):
    """ Continuous column pdf given by the normal distribution. """

    name = "Normal"
    dtype = "float"
    mean_limits = [-10, 10]
    std_limits = [0, 10]

    def __init__(self):

        self.mean = np.random.uniform(*self.mean_limits)
        self.std = np.random.uniform(*self.std_limits)

    def sample(self, nrows):
        """ Take a sample of size `nrows` from the normal distribution with
        mean and standard deviation given by `mean` and `std` respectively. """

        return np.random.normal(loc=self.mean, scale=self.std, size=nrows)


# ======================
# DISCRETE DISTRIBUTIONS
# ======================


class Bernoulli(Distribution):
    """ Discrete column pdf given by the Bernoulli distribution. That is, the
    binomial distribution with 1 trial. """

    name = "Bernoulli"
    dtype = "int"
    prob_limits = [0, 1]

    def __init__(self):

        self.prob = np.random.uniform(*self.prob_limits)

    def sample(self, nrows):
        """ Take a sample of size `nrows` from the Bernoulli distribution with
        parameter `prob`. """

        return np.random.binomial(n=1, p=self.prob, size=nrows)


class Poisson(Distribution):
    """ Discrete column pdf given by the Poisson distribution. """

    name = "Poisson"
    dtype = "int"
    lam_limits = [0, 10]

    def __init__(self):

        self.lam = np.random.uniform(*self.lam_limits)

    def sample(self, nrows):
        """ Take a sample of size `nrows` from the Poisson distribution with
        parameter `lam`. """

        return np.random.poisson(lam=self.lam, size=nrows)
