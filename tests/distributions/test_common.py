""" Test base distribution class and the methods shared by all pdf classes. """

import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import integers, sampled_from

from edo.distributions import Distribution, all_distributions


def test_distribution_instantiation():
    """ Verify Distribution cannot be instantiated. """

    with pytest.raises(TypeError):
        Distribution()


params = given(
    distribution=sampled_from(all_distributions),
    seed=integers(min_value=0, max_value=1000),
)


@params
def test_init(distribution, seed):
    """ Check defaults of distribution objects. """

    state = np.random.RandomState(seed)
    pdf = distribution(state)

    assert pdf.name == distribution.name
    assert pdf.param_limits == distribution.param_limits

    for name, value in vars(pdf).items():
        limits = pdf.param_limits[name]
        try:
            for val in value:
                assert min(limits) <= val <= max(limits)
        except TypeError:
            assert min(limits) <= value <= max(limits)


@params
def test_repr(distribution, seed):
    """ Assert that distribution objects have the correct string. """

    state = np.random.RandomState(seed)
    pdf = distribution(state)

    assert str(pdf).startswith(pdf.name)
    for name in vars(pdf):
        assert name in str(pdf)


@params
def test_set_param_limits(distribution, seed):
    """ Check distribution classes can have their default parameter limits
    changed. """

    param_limits = dict(distribution.param_limits)
    for param_name in distribution.param_limits:
        distribution.param_limits[param_name] = None

    assert distribution.param_limits != param_limits
    distribution.param_limits = param_limits


@given(
    distribution=sampled_from(all_distributions),
    nrows=integers(min_value=1, max_value=100),
    seed=integers(min_value=0, max_value=100),
)
def test_sample(distribution, nrows, seed):
    """ Verify that distribution objects can sample correctly. """

    state = np.random.RandomState(seed)
    pdf = distribution(state)

    sample = pdf.sample(nrows, state)
    assert sample.shape == (nrows,)
    assert sample.dtype == pdf.dtype
