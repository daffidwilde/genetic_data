""" Tests for the Family subtype-handler class. """

import os
import pathlib

from hypothesis import given
from hypothesis.strategies import composite, sampled_from

from edo import Family
from edo.distributions import all_distributions


@composite
def distributions(draw, pool=all_distributions):
    """ Draw a distribution from the pool. """

    return draw(sampled_from(pool))


@given(distribution=distributions())
def test_init(distribution):
    """ Test that a Family object can be instantiated correctly. """

    family = Family(distribution)

    assert family.distribution is distribution
    assert family.max_subtypes is None
    assert family.name == distribution.name + "Family"
    assert family.subtype_id == 0
    assert family.subtypes == {}


@given(distribution=distributions())
def test_repr(distribution):
    """ Test that the string representation of a Family object is correct. """

    family = Family(distribution)

    assert repr(family).startswith(family.name)
    assert str(family.subtype_id) in repr(family)


@given(distribution=distributions())
def test_add_subtype(distribution):
    """ Test that a new subtype can be created correctly. """

    family = Family(distribution)
    family.add_subtype()

    subtype = family.subtypes.get(0)

    assert family.subtype_id == 1
    assert issubclass(subtype, distribution)
    assert subtype.__name__ == f"{distribution.name}Subtype"
    assert subtype.subtype_id == 0


@given(distribution=distributions())
def test_make_instance(distribution):
    """ Test that an instance can be created correctly. """

    family = Family(distribution)
    pdf = family.make_instance()

    assert family.subtype_id == 1
    assert family.subtypes == {0: pdf.__class__}
    assert isinstance(pdf, distribution)
    assert isinstance(pdf, family.subtypes[0])


@given(distribution=distributions())
def test_save(distribution):
    """ Test that a family can save its subtypes correctly. """

    family = Family(distribution)
    family.add_subtype()
    family.save(".testcache")

    path = pathlib.Path(f".testcache/subtypes/{distribution.name}/0.pkl")
    assert path.exists()

    os.system("rm -r .testcache")


@given(distribution=distributions())
def test_reset(distribution):
    """ Test that a family can reset itself. """

    family = Family(distribution)
    family.add_subtype()
    family.reset()

    assert family.subtype_id == 0
    assert family.subtypes == {}


@given(distribution=distributions())
def test_reset_cached(distribution):
    """ Test that a family can remove any cached subtypes. """

    family = Family(distribution)
    family.add_subtype()
    family.save(".testcache")
    family.reset(".testcache")

    path = pathlib.Path(f".testcache/subtypes/{distribution.name}")
    assert not path.exists()


@given(distribution=distributions())
def test_load(distribution):
    """ Test that a family can be created from a cache. """

    family = Family(distribution)
    family.add_subtype()
    subtype = family.subtypes[0]
    family.save(".testcache")

    pickled = Family.load(distribution, cache_dir=".testcache")
    pickled_subtype = pickled.subtypes[0]

    assert isinstance(pickled, Family)
    assert pickled.distribution is distribution
    assert pickled.subtype_id == 1
    assert pickled.subtypes == {0: pickled_subtype}

    assert issubclass(pickled_subtype, distribution)
    assert pickled_subtype.__name__ == subtype.__name__
    assert pickled_subtype.name == subtype.name
    assert pickled_subtype.dtype == subtype.dtype
    assert pickled_subtype.hard_limits == subtype.hard_limits
    assert pickled_subtype.param_limits == subtype.param_limits
    assert pickled_subtype.__init__ is subtype.__init__
    assert pickled_subtype.__repr__ is subtype.__repr__
    assert pickled_subtype.sample is subtype.sample

    os.system("rm -r .testcache")
