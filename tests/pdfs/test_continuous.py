""" Continuous pdf tests. """

import numpy as np
from hypothesis import given
from hypothesis.strategies import floats, integers, tuples

from edo.pdfs.continuous import Gamma, Normal, Uniform

LIMITS = (
    tuples(floats(min_value=0, max_value=10), floats(min_value=0, max_value=10))
    .map(sorted)
    .filter(lambda x: x[0] <= x[1])
)

CONTINUOUS = given(
    first_limits=LIMITS,
    second_limits=LIMITS,
    seed=integers(min_value=0, max_value=2 ** 32 - 1),
)


@CONTINUOUS
def test_gamma_set_param_limits(first_limits, second_limits, seed):
    """ Check that a Gamma object can sample its parameters correctly if its
    class attributes are altered. """

    Gamma.param_limits = {"alpha": first_limits, "theta": second_limits}

    np.random.seed(seed)
    gamma = Gamma()
    assert gamma.alpha >= first_limits[0] and gamma.alpha <= first_limits[1]
    assert gamma.theta >= second_limits[0] and gamma.theta <= second_limits[1]


@CONTINUOUS
def test_normal_set_param_limits(first_limits, second_limits, seed):
    """ Check that a Normal object can sample its parameters correctly if its
    class attributes are altered. """

    Normal.param_limits = {"mean": first_limits, "std": second_limits}

    np.random.seed(seed)
    normal = Normal()
    assert normal.mean >= first_limits[0] and normal.mean <= first_limits[1]
    assert normal.std >= second_limits[0] and normal.std <= second_limits[1]

@CONTINUOUS
def test_uniform_set_param_limits(first_limits, second_limits, seed):
    """ Check that a Uniform object can sample its parameters correctly if its
    class attributes are altered. """

    Uniform.param_limits = {"bounds": first_limits}

    np.random.seed(seed)
    uniform = Uniform()
    for bound in uniform.bounds:
        assert bound >= first_limits[0]
        assert bound <= first_limits[1]
