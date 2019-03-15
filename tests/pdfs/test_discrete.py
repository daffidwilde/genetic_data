""" Discrete pdf tests. """

import numpy as np
from hypothesis import given
from hypothesis.strategies import floats, integers, tuples

from edo.pdfs.discrete import Bernoulli, Poisson

LIMITS = (
    tuples(floats(min_value=0, max_value=10), floats(min_value=0, max_value=10))
    .map(sorted)
    .filter(lambda x: x[0] <= x[1])
)


@given(
    prob_limits=tuples(
        floats(min_value=0, max_value=1), floats(min_value=0, max_value=1)
    )
    .map(sorted)
    .filter(lambda x: x[0] < x[1]),
    seed=integers(min_value=0, max_value=2 ** 32 - 1),
)
def test_bernoulli_set_param_limits(prob_limits, seed):
    """ Check that a Bernoulli object can sample its parameters correctly if its
    class attributes are altered. """

    Bernoulli.param_limits = {"prob": prob_limits}

    np.random.seed(seed)
    bernoulli = Bernoulli()
    assert prob_limits[0] <= bernoulli.prob <= prob_limits[1]


@given(lam_limits=LIMITS, seed=integers(min_value=0, max_value=2 ** 32 - 1))
def test_poisson_set_param_limits(lam_limits, seed):
    """ Check that a Poisson object can sample its parameters correctly if its
    class attributes are altered. """

    Poisson.param_limits = {"lam": lam_limits}

    np.random.seed(seed)
    poisson = Poisson()
    assert lam_limits[0] <= poisson.lam <= lam_limits[1]
