""" Test base distribution class and the methods shared by all pdf classes. """

# pylint: disable=protected-access

import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import integers

from edo.pdfs import Distribution, all_pdfs


def test_distribution_init():
    """ Verify defaults of Distribution instance. """

    dist = Distribution()
    assert dist.name == "Distribution"
    assert dist.param_limits is None

    with pytest.raises(NotImplementedError):
        dist.sample()


@given(seed=integers(min_value=0, max_value=2 ** 32 - 1))
def test_init(seed):
    """ Check defaults of distribution objects. """

    np.random.seed(seed)
    for pdf_class in all_pdfs:
        pdf = pdf_class()
        assert pdf.name == pdf_class.name
        assert pdf.param_limits == pdf_class.param_limits

        for name, value in vars(pdf).items():
            limits = pdf.param_limits[name]
            if isinstance(value, list):
                for val in value:
                    assert val >= min(limits)
                    assert val <= max(limits)
            else:
                assert value >= min(limits)
                assert value <= max(limits)


@given(seed=integers(min_value=0, max_value=2 ** 32 - 1))
def test_repr(seed):
    """ Assert that distribution objects have the correct string. """

    np.random.seed(seed)
    for pdf_class in all_pdfs:
        pdf = pdf_class()
        assert str(pdf).startswith(pdf.name)


@given(
    nrows=integers(min_value=1, max_value=1000),
    seed=integers(min_value=0, max_value=2 ** 32 - 1),
)
def test_sample(nrows, seed):
    """ Verify that distribution objects can sample correctly. """

    np.random.seed(seed)
    for pdf_class in all_pdfs:
        pdf = pdf_class()
        sample = pdf.sample(nrows)
        assert sample.shape == (nrows,)
        assert sample.dtype == pdf.dtype


@given(seed=integers(min_value=0, max_value=2 ** 32 - 1))
def test_to_tuple(seed):
    """ Verify that objects can pass their information to a tuple of the correct
    length and form. """

    np.random.seed(seed)
    for pdf_class in all_pdfs:
        pdf = pdf_class()
        out = pdf.to_tuple()
        assert len(out) - 2 * len(pdf.__dict__) == 1
        assert out[0] == pdf.name
        for i, item in enumerate(out[1:]):
            if i % 2 == 0:
                assert isinstance(item, str)
            else:
                assert item == list(pdf.__dict__.values())[int((i - 1) / 2)]


def test_reset():
    """ Check distribution classes can be reset to default parameter limits. """

    for pdf_class in all_pdfs:
        pdf_class()
        for param_name in pdf_class.param_limits:
            pdf_class.param_limits[param_name] = None

        assert pdf_class.param_limits != pdf_class._param_limits

        pdf_class.reset()
        assert pdf_class.param_limits == pdf_class._param_limits
