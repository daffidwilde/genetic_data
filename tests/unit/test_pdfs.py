""" Unit tests for the standard columns pdf's. """

import numpy as np
from hypothesis import given
from hypothesis.strategies import floats, integers, tuples
from genetic_data.pdfs import Gamma, Normal, Poisson

LIMITS = (
    tuples(floats(min_value=0, max_value=10), floats(min_value=0, max_value=10))
    .map(sorted)
    .filter(lambda x: x[0] <= x[1])
)


@given(nrows=integers(min_value=1), seed=integers(min_value=0))
def test_gamma_sample(nrows, seed):
    """ Verify that a Gamma object can sample correctly. """
    np.random.seed(seed)
    gamma = Gamma()
    sample = gamma.sample(nrows)
    assert sample.shape == (nrows,)
    assert sample.dtype == "float"


@given(alpha_limits=LIMITS, theta_limits=LIMITS, seed=integers(min_value=0))
def test_gamma_set_param_limits(alpha_limits, theta_limits, seed):
    """ Check that a Gamma object can sample its parameters correctly if its
    class attributes are altered. """

    Gamma.alpha_limits = alpha_limits
    Gamma.theta_limits = theta_limits

    np.random.seed(seed)
    gamma = Gamma()
    assert gamma.alpha >= alpha_limits[0] and gamma.alpha <= alpha_limits[1]
    assert gamma.theta >= theta_limits[0] and gamma.theta <= theta_limits[1]


@given(nrows=integers(min_value=1), seed=integers(min_value=0))
def test_normal_sample(nrows, seed):
    """ Verify that a Normal object can sample correctly. """
    np.random.seed(seed)
    normal = Normal()
    sample = normal.sample(nrows)
    assert sample.shape == (nrows,)
    assert sample.dtype == "float"


@given(mean_limits=LIMITS, std_limits=LIMITS, seed=integers(min_value=0))
def test_normal_set_param_limits(mean_limits, std_limits, seed):
    """ Check that a Normal object can sample its parameters correctly if its
    class attributes are altered. """

    Normal.mean_limits = mean_limits
    Normal.std_limits = std_limits

    np.random.seed(seed)
    normal = Normal()
    assert normal.mean >= mean_limits[0] and normal.mean <= mean_limits[1]
    assert normal.std >= std_limits[0] and normal.std <= std_limits[1]


@given(nrows=integers(min_value=1), seed=integers(min_value=0))
def test_poisson_sample(nrows, seed):
    """ Verify that a Poisson object can sample correctly. """
    np.random.seed(seed)
    poisson = Poisson()
    sample = poisson.sample(nrows)
    assert sample.shape == (nrows,)
    assert sample.dtype == "int"


@given(lam_limits=LIMITS, seed=integers(min_value=0))
def test_poisson_set_param_limits(lam_limits, seed):
    """ Check that a Poisson object can sample its parameters correctly if its
    class attributes are altered. """

    Poisson.lam_limits = lam_limits

    np.random.seed(seed)
    poisson = Poisson()
    assert poisson.lam >= lam_limits[0] and poisson.lam <= lam_limits[1]
