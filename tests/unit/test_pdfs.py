""" Unit tests for the standard columns pdf's. """

import numpy as np

import pytest

from hypothesis import given
from hypothesis.strategies import floats, integers, tuples

from genetic_data.pdfs import Distribution, Gamma, Normal, Bernoulli, Poisson


LIMITS = (
    tuples(floats(min_value=0, max_value=10), floats(min_value=0, max_value=10))
    .map(sorted)
    .filter(lambda x: x[0] <= x[1])
)


def test_Distribution_sample():
    """ Verify Distribution object alone raises an error when trying to sample
    from it. """

    with pytest.raises(NotImplementedError):
        dist = Distribution()
        sample = dist.sample()


def test_to_tuple():
    """ Verify that objects can pass their information to a tuple of the correct
    length and form. """

    for pdf_class in [Gamma, Normal, Bernoulli, Poisson]:
        pdf = pdf_class()
        out = pdf.to_tuple()
        assert 2 * len(pdf.__dict__) - len(out) == 1
        assert out[0] == pdf.name


# =====
# GAMMA
# =====


@given(seed=integers(min_value=0))
def test_Gamma_string(seed):
    """ Assert that a Gamma object has the correct string presentation. """
    np.random.seed(0)
    gamma = Gamma()
    assert str(gamma).startswith("Gamma")


@given(nrows=integers(min_value=1), seed=integers(min_value=0))
def test_Gamma_sample(nrows, seed):
    """ Verify that a Gamma object can sample correctly. """
    np.random.seed(seed)
    gamma = Gamma()
    sample = gamma.sample(nrows)
    assert sample.shape == (nrows,)
    assert sample.dtype == "float"


@given(alpha_limits=LIMITS, theta_limits=LIMITS, seed=integers(min_value=0))
def test_Gamma_set_param_limits(alpha_limits, theta_limits, seed):
    """ Check that a Gamma object can sample its parameters correctly if its
    class attributes are altered. """

    Gamma.alpha_limits = alpha_limits
    Gamma.theta_limits = theta_limits

    np.random.seed(seed)
    gamma = Gamma()
    assert gamma.alpha >= alpha_limits[0] and gamma.alpha <= alpha_limits[1]
    assert gamma.theta >= theta_limits[0] and gamma.theta <= theta_limits[1]


# ======
# NORMAL
# ======


@given(seed=integers(min_value=0))
def test_Normal_string(seed):
    """ Assert that a Normal object has the correct string representation. """
    np.random.seed(0)
    normal = Normal()
    assert str(normal).startswith("Normal")


@given(nrows=integers(min_value=1), seed=integers(min_value=0))
def test_Normal_sample(nrows, seed):
    """ Verify that a Normal object can sample correctly. """
    np.random.seed(seed)
    normal = Normal()
    sample = normal.sample(nrows)
    assert sample.shape == (nrows,)
    assert sample.dtype == "float"


@given(mean_limits=LIMITS, std_limits=LIMITS, seed=integers(min_value=0))
def test_Normal_set_param_limits(mean_limits, std_limits, seed):
    """ Check that a Normal object can sample its parameters correctly if its
    class attributes are altered. """

    Normal.mean_limits = mean_limits
    Normal.std_limits = std_limits

    np.random.seed(seed)
    normal = Normal()
    assert normal.mean >= mean_limits[0] and normal.mean <= mean_limits[1]
    assert normal.std >= std_limits[0] and normal.std <= std_limits[1]


# =========
# BERNOULLI
# =========


@given(seed=integers(min_value=0))
def test_Bernoulli_string(seed):
    """ Assert that a Poisson object has the correct string representation. """
    np.random.seed(seed)
    bernoulli = Bernoulli()
    assert str(bernoulli).startswith("Bernoulli")


@given(nrows=integers(min_value=1), seed=integers(min_value=0))
def test_Bernoulli_sample(nrows, seed):
    np.random.seed(seed)
    bernoulli = Bernoulli()
    sample = bernoulli.sample(nrows)
    assert sample.shape == (nrows,)
    assert sample.dtype == "int"


@given(
    prob_limits=tuples(
        floats(min_value=0, max_value=1), floats(min_value=0, max_value=1)
    )
    .map(sorted)
    .filter(lambda x: x[0] < x[1]),
    seed=integers(min_value=0),
)
def test_Bernoulli_set_param_limits(prob_limits, seed):
    """ Check that a Bernoulli object can sample its parameters correctly if its
    class attributes are altered. """

    Bernoulli.prob_limits = prob_limits

    np.random.seed(seed)
    bernoulli = Bernoulli()
    assert bernoulli.prob >= prob_limits[0] and bernoulli.prob <= prob_limits[1]


# =======
# POISSON
# =======


@given(seed=integers(min_value=0))
def test_Poisson_string(seed):
    """ Assert that a Poisson object has the correct string representation. """
    np.random.seed(seed)
    poisson = Poisson()
    assert str(poisson).startswith("Poisson")


@given(nrows=integers(min_value=1), seed=integers(min_value=0))
def test_Poisson_sample(nrows, seed):
    """ Verify that a Poisson object can sample correctly. """
    np.random.seed(seed)
    poisson = Poisson()
    sample = poisson.sample(nrows)
    assert sample.shape == (nrows,)
    assert sample.dtype == "int"


@given(lam_limits=LIMITS, seed=integers(min_value=0))
def test_Poisson_set_param_limits(lam_limits, seed):
    """ Check that a Poisson object can sample its parameters correctly if its
    class attributes are altered. """

    Poisson.lam_limits = lam_limits

    np.random.seed(seed)
    poisson = Poisson()
    assert poisson.lam >= lam_limits[0] and poisson.lam <= lam_limits[1]
