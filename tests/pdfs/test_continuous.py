""" Continuous pdf tests. """

import numpy as np
from hypothesis import given
from hypothesis.strategies import floats, integers, tuples

from edo.pdfs.continuous import Gamma, Normal

LIMITS = (
    tuples(floats(min_value=0, max_value=10), floats(min_value=0, max_value=10))
    .map(sorted)
    .filter(lambda x: x[0] <= x[1])
)

CONTINUOUS = given(
    alpha_limits=LIMITS,
    theta_limits=LIMITS,
    seed=integers(min_value=0, max_value=2 ** 32 - 1),
)


@CONTINUOUS
def test_gamma_set_param_limits(alpha_limits, theta_limits, seed):
    """ Check that a Gamma object can sample its parameters correctly if its
    class attributes are altered. """

    Gamma.param_limits = {"alpha": alpha_limits, "theta": theta_limits}

    np.random.seed(seed)
    gamma = Gamma()
    assert gamma.alpha >= alpha_limits[0] and gamma.alpha <= alpha_limits[1]
    assert gamma.theta >= theta_limits[0] and gamma.theta <= theta_limits[1]


@CONTINUOUS
def test_normal_set_param_limits(mean_limits, std_limits, seed):
    """ Check that a Normal object can sample its parameters correctly if its
    class attributes are altered. """

    Normal.param_limits = {"mean": mean_limits, "std": std_limits}

    np.random.seed(seed)
    normal = Normal()
    assert normal.mean >= mean_limits[0] and normal.mean <= mean_limits[1]
    assert normal.std >= std_limits[0] and normal.std <= std_limits[1]
