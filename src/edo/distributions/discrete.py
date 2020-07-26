""" All currently implemented discrete distribution classes. """

from .base import Distribution


class Bernoulli(Distribution):
    """ The Bernoulli distribution class, i.e. a binomial distribution with
    exactly one trial.

    Parameters
    ----------
    random_state : numpy.random.RandomState
        The PRNG used to sample instance parameters from ``param_limits``.

    Attributes
    ----------
    name : str
        Name of the distribution, ``"Bernoulli"``.
    dtype : int
        Preferred datatype of distribution, ``int``.
    param_limits : dict
        A dictionary of the limits on the distribution parameter. Defaults to
        ``[0, 1]`` for ``prob``.
    prob : float
        The success probability, sampled from ``param_limits["prob"]``.
        `Instance attribute.`
    """

    name = "Bernoulli"
    dtype = int
    hard_limits = {"prob": [0, 1]}
    param_limits = {"prob": [0, 1]}

    def __init__(self, random_state):

        self.prob = random_state.uniform(*self.param_limits["prob"])

    def __repr__(self):

        return f"Bernoulli(prob={round(self.prob, 2)})"

    def sample(self, nrows, random_state):
        """ Take a sample of size ``nrows`` from the Bernoulli distribution
        using the provided ``np.random.RandomState`` instance. """

        return random_state.binomial(n=1, p=self.prob, size=nrows)


class Poisson(Distribution):
    """ The Poisson distribution class.

    Parameters
    ----------
    random_state : numpy.random.RandomState
        The PRNG used to sample instance parameters from ``param_limits``.

    Attributes
    ----------
    name : str
        Name of distribution, ``"Poisson"``.
    dtype : int
        Preferred datatype of distribution, ``int``.
    param_limits : dict
        A dictionary of the limits of the distribution parameter. Defaults to
        ``[0, 10]`` for ``lam``.
    lam : float
        The rate parameter, sampled from ``param_limits["lam"]``.
        `Instance attribute.`
    """

    name = "Poisson"
    dtype = int
    hard_limits = {"lam": [0, 10]}
    param_limits = {"lam": [0, 10]}

    def __init__(self, random_state):

        self.lam = random_state.uniform(*self.param_limits["lam"])

    def __repr__(self):

        return f"Poisson(lam={round(self.lam, 2)})"

    def sample(self, nrows, random_state):
        """ Take a sample of size ``nrows`` from the Poisson distribution
        using the provided ``np.random.RandomState`` instance. """

        return random_state.poisson(lam=self.lam, size=nrows)
