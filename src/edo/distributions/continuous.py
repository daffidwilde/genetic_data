""" All currently implemented continuous distributions. """

from .base import Distribution


class Gamma(Distribution):
    """ The gamma distribution class.

    Parameters
    ----------
    random_state : numpy.random.RandomState
        The PRNG used to sample instance parameters from ``param_limits``.

    Attributes
    ----------
    name : str
        Name of the distribution, ``"Gamma"``.
    dtype : float
        Preferred datatype of distribution, ``float``.
    param_limits : dict
        A dictionary of limits on the distribution parameters. Defaults to
        ``[0, 10]`` for both ``alpha`` and ``theta``.
    alpha : float
        The shape parameter sampled from ``param_limits["alpha"]``.
        `Instance attribute.`
    theta : float
        The scale parameter sampled from ``param_limits["theta"]``.
        `Instance attribute.`
    """

    name = "Gamma"
    dtype = float
    hard_limits = {"alpha": [0, 10], "theta": [0, 10]}
    param_limits = {"alpha": [0, 10], "theta": [0, 10]}

    def __init__(self, random_state):

        self.alpha = random_state.uniform(*self.param_limits["alpha"])
        self.theta = random_state.uniform(*self.param_limits["theta"])

    def __repr__(self):

        alpha = round(self.alpha, 2)
        theta = round(self.theta, 2)
        return f"Gamma(alpha={alpha}, theta={theta})"

    def sample(self, nrows, random_state):
        """ Take a sample of size ``nrows`` from the gamma distribution using
        the provided ``np.random.RandomState`` instance. """

        return random_state.gamma(
            shape=self.alpha, scale=self.theta, size=nrows
        )


class Normal(Distribution):
    """ The normal distribution class.

    Parameters
    ----------
    random_state : numpy.random.RandomState
        The PRNG used to sample instance parameters from ``param_limits``.

    Attributes
    ----------
    name : str
        Name of the distribution, ``"Normal"``.
    dtype : float
        Preferred datatype of distribution, ``float``.
    param_limits : dict
        A dictionary of limits on the distribution parameters. Defaults to
        ``[-10, 10]`` for ``mean`` and ``[0, 10]`` for ``std``.
    mean : float"
        The mean, sampled from ``param_limits["mean"]``. `Instance attribute.`
    std : float
        The standard deviation, sampled from ``param_limits["std"]``.
        `Instance attribute.`
    """

    name = "Normal"
    dtype = float
    hard_limits = {"mean": [-10, 10], "std": [0, 10]}
    param_limits = {"mean": [-10, 10], "std": [0, 10]}

    def __init__(self, random_state):

        self.mean = random_state.uniform(*self.param_limits["mean"])
        self.std = random_state.uniform(*self.param_limits["std"])

    def __repr__(self):

        mean = round(self.mean, 2)
        std = round(self.std, 2)
        return f"Normal(mean={mean}, std={std})"

    def sample(self, nrows, random_state):
        """ Take a sample of size ``nrows`` from the normal distribution using
        the provided ``np.random.RandomState`` instance. """

        return random_state.normal(loc=self.mean, scale=self.std, size=nrows)


class Uniform(Distribution):
    """ The uniform distribution class.

    Parameters
    ----------
    random_state : numpy.random.RandomState
        The PRNG used to sample instance parameters from ``param_limits``.

    Attributes
    ----------
    name : str
        Name of the distribution, ``Uniform``.
    dtype : float
        Preferred datatype of distribution, ``float``.
    param_limits : dict
        A dictionary of limits on the distribution parameters. Defaults to
        ``[-10, 10]`` for ``bounds``.
    bounds : list of float
        The lower and upper bounds of the distribution. `Instance attribute`.
    """

    name = "Uniform"
    dtype = float
    hard_limits = {"bounds": [-10, 10]}
    param_limits = {"bounds": [-10, 10]}

    def __init__(self, random_state):

        self.bounds = sorted(
            random_state.uniform(*self.param_limits["bounds"], size=2)
        )

    def __repr__(self):

        lower = round(min(self.bounds), 2)
        upper = round(max(self.bounds), 2)
        return f"Uniform(bounds=[{lower}, {upper}])"

    def sample(self, nrows, random_state):
        """ Take a sample of size ``nrows`` from the uniform distribution using
        the provided ``np.random.RandomState`` instance. """

        return random_state.uniform(*self.bounds, size=nrows)
