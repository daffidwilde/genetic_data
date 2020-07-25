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


@given(distribution=sampled_from(all_distributions))
def test_init(distribution):
    """ Check defaults of distribution objects. """

    pdf = distribution()
    assert pdf.name == distribution.name
    assert pdf.param_limits == distribution.param_limits

    for name, value in vars(pdf).items():
        limits = pdf.param_limits[name]
        try:
            for val in value:
                assert min(limits) <= val <= max(limits)
        except TypeError:
            assert min(limits) <= value <= max(limits)


@given(distribution=sampled_from(all_distributions))
def test_repr(distribution):
    """ Assert that distribution objects have the correct string. """

    pdf = distribution()
    assert str(pdf).startswith(pdf.name)
    for name in vars(pdf):
        assert name in str(pdf)


@given(
    distribution=sampled_from(all_distributions),
    nrows=integers(min_value=1, max_value=100),
    seed=integers(min_value=0, max_value=100),
)
def test_sample(distribution, nrows, seed):
    """ Verify that distribution objects can sample correctly. """

    pdf = distribution()
    state = np.random.RandomState(seed)

    sample = pdf.sample(nrows, state)
    assert sample.shape == (nrows,)
    assert sample.dtype == pdf.dtype


@given(distribution=sampled_from(all_distributions))
def test_set_param_limits(distribution):
    """ Check distribution classes can have their default parameter limits
    changed. """

    param_limits = dict(distribution.param_limits)
    for param_name in distribution.param_limits:
        distribution.param_limits[param_name] = None

    assert distribution.param_limits != param_limits
    distribution.param_limits = param_limits
