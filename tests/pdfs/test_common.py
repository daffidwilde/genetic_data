""" Test base distribution class and the methods shared by all pdf classes. """

import pytest
from hypothesis import given
from hypothesis.strategies import integers, sampled_from

from edo.pdfs import Distribution, all_pdfs


def test_distribution_init():
    """ Verify defaults of Distribution instance. """

    dist = Distribution()
    assert dist.name == "Distribution"
    assert dist.subtype_id == 0
    assert dist.subtypes == []
    assert dist.max_subtypes is None
    assert dist.param_limits is None

    with pytest.raises(NotImplementedError):
        dist.sample()


@given(family=sampled_from(all_pdfs))
def test_init(family):
    """ Check defaults of distribution objects. """

    pdf = family()
    assert pdf.name == family.name
    assert pdf.param_limits == family.param_limits

    for name, value in vars(pdf).items():
        limits = pdf.param_limits[name]
        try:
            for val in value:
                assert min(limits) <= val <= max(limits)
        except TypeError:
            assert min(limits) <= value <= max(limits)


@given(family=sampled_from(all_pdfs))
def test_repr(family):
    """ Assert that distribution objects have the correct string. """

    pdf = family()
    assert str(pdf).startswith(pdf.name)


@given(family=sampled_from(all_pdfs))
def test_build_subtype(family):
    """ Test that a distribution family can make a subtype (copy) of itself. """

    family.reset()
    subtype = family.build_subtype()
    for key, value in vars(subtype).items():
        if key in vars(family) and "subtype" not in key:
            assert getattr(family, key) == value

    assert subtype.__repr__ is family.__repr__
    assert subtype.sample is family.sample
    assert subtype.to_dict is family.to_dict
    assert subtype.subtype_id == 0
    assert family.subtype_id == 1

    sub = subtype()
    assert sub.family == family
    assert isinstance(sub, subtype)
    assert not isinstance(sub, family)


@given(family=sampled_from(all_pdfs))
def test_make_instance(family):
    """ Test that distribution objects can make instances in its family
    correctly. """

    family.reset()
    pdf = family.make_instance()

    assert family.subtypes == [pdf.__class__]
    assert isinstance(pdf, family.subtypes[0])

    pdf = family.make_instance()
    if len(family.subtypes) == 1:
        assert isinstance(pdf, family.subtypes[0])
    else:
        assert len(family.subtypes) == 2
        assert isinstance(pdf, family.subtypes[1])


@given(family=sampled_from(all_pdfs))
def test_max_subtypes(family):
    """ Test that distribution objects cannot make more subtypes than their
    prescribed maximum. """

    family.reset()

    family.max_subtypes = 0
    with pytest.raises(ValueError):
        family.make_instance()

    family.max_subtypes = 1
    family.make_instance()
    subtype = family.subtypes[0]
    pdf = family.make_instance()
    assert isinstance(pdf, subtype)


@given(family=sampled_from(all_pdfs))
def test_reset(family):
    """ Test that distribution classes can be reset. """

    family.subtype_id = 1
    family.subtypes = ["foo"]
    family.reset()
    assert family.subtype_id == 0
    assert family.subtypes == []


@given(family=sampled_from(all_pdfs), nrows=integers(min_value=0, max_value=99))
def test_sample(family, nrows):
    """ Verify that distribution objects can sample correctly. """

    pdf = family()
    sample = pdf.sample(nrows)
    assert sample.shape == (nrows,)
    assert sample.dtype == pdf.dtype


@given(family=sampled_from(all_pdfs))
def test_to_dict(family):
    """ Verify that objects can pass their information to a dictionary of the
    correct form. """

    pdf = family()
    pdf_dict = pdf.to_dict()
    assert pdf_dict["name"] == pdf.name
    assert pdf_dict["subtype_id"] == pdf.subtype_id

    for param in pdf.param_limits:
        assert pdf_dict[param] == vars(pdf)[param]


@given(family=sampled_from(all_pdfs))
def test_set_param_limits(family):
    """ Check distribution classes can have their default parameter limits
    changed. """

    param_limits = dict(family.param_limits)
    for param_name in family.param_limits:
        family.param_limits[param_name] = None

    assert family.param_limits != param_limits
    family.param_limits = param_limits
