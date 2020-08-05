""" Tests for the Family subtype-handler class. """

import os
import pathlib

import numpy as np
from hypothesis import given, settings
from hypothesis.strategies import composite, integers, sampled_from

from edo import Family
from edo.distributions import all_distributions


@composite
def distributions(draw, pool=all_distributions):
    """ Draw a distribution from the pool. """

    return draw(sampled_from(pool))


@composite
def states(draw, min_value=0, max_value=1000):
    """ Create an instance of `np.random.RandomState`. """

    seed = draw(integers(min_value, max_value))
    return np.random.RandomState(seed)


@given(distribution=distributions())
def test_init(distribution):
    """ Test that a Family object can be instantiated correctly. """

    family = Family(distribution)

    assert family.distribution is distribution
    assert family.max_subtypes is None
    assert family.name == distribution.name + "Family"
    assert family.subtype_id == 0
    assert family.subtypes == {}
    assert family.all_subtypes == {}
    assert family.random_state is np.random.mtrand._rand


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
    assert subtype.family is family
    assert subtype is family.all_subtypes.get(0)


@given(distribution=distributions(), state=states())
def test_make_instance(distribution, state):
    """ Test that an instance can be created correctly. """

    family = Family(distribution)
    pdf = family.make_instance(state)

    assert family.subtype_id == 1
    assert family.subtypes == {0: pdf.__class__}
    assert isinstance(pdf, distribution)
    assert isinstance(pdf, family.subtypes[0])
    assert pdf.family is family


@given(distribution=distributions())
def test_keep_track_all_subtypes(distribution):
    """ Test that a family can keep track of all of its subtypes. """

    family = Family(distribution)
    family.add_subtype()
    family.add_subtype()

    subtype0, subtype1 = family.subtypes.values()

    family.subtypes.pop(0)

    assert family.subtypes == {1: subtype1}
    assert family.all_subtypes == {0: subtype0, 1: subtype1}


@given(distribution=distributions())
@settings(deadline=None)
def test_save(distribution):
    """ Test that a family can save its subtypes correctly. """

    family = Family(distribution)
    family.add_subtype()
    family.save(".testcache")

    path = pathlib.Path(f".testcache/subtypes/{distribution.name}/")
    assert (path / "0.pkl").exists()
    assert (path / "state.pkl").exists()

    os.system("rm -r .testcache")


@given(distribution=distributions())
def test_reset(distribution):
    """ Test that a family can reset itself. """

    family = Family(distribution)
    family.add_subtype()
    family.reset()

    fresh_family = Family(distribution)

    assert vars(family) == vars(fresh_family)


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

    pickled = Family.load(distribution, root=".testcache")
    pickled_subtype = pickled.subtypes[0]

    assert isinstance(pickled, Family)
    assert pickled.distribution is distribution
    assert pickled.subtype_id == 1
    assert pickled.subtypes == {0: pickled_subtype}

    assert issubclass(pickled_subtype, distribution)
    assert pickled_subtype.__name__ == subtype.__name__
    assert pickled_subtype.name == subtype.name
    assert pickled_subtype.dtype == subtype.dtype
    assert pickled_subtype.subtype_id == 0
    assert pickled_subtype.family is pickled

    assert pickled_subtype.hard_limits == subtype.hard_limits
    assert pickled_subtype.param_limits == subtype.param_limits
    assert pickled_subtype.__init__ is subtype.__init__
    assert pickled_subtype.__repr__ is subtype.__repr__
    assert pickled_subtype.sample is subtype.sample

    for fpart, ppart in zip(
        family.random_state.get_state(), pickled.random_state.get_state()
    ):
        try:
            assert all(fpart == ppart)
        except TypeError:
            assert fpart == ppart

    os.system("rm -r .testcache")


@given(distribution=distributions())
@settings(deadline=None)
def test_load_more_than_ten(distribution):
    """ Test that a family with more than 10 subtypes can be created from a
    cache. """

    family = Family(distribution)
    for _ in range(11):
        family.add_subtype()

    family.save(".testcache")

    pickled = Family.load(distribution, root=".testcache")

    assert isinstance(pickled, Family)
    assert pickled.distribution is distribution
    assert pickled.subtype_id == 11
    assert list(pickled.subtypes.keys()) == list(range(11))

    for subtype_id, subtype in pickled.subtypes.items():
        assert issubclass(subtype, distribution)
        assert subtype.__init__ is distribution.__init__
        assert subtype.sample is distribution.sample
        assert subtype.subtype_id == subtype_id
        assert subtype.family is pickled

    for fpart, ppart in zip(
        family.random_state.get_state(), pickled.random_state.get_state()
    ):
        try:
            assert all(fpart == ppart)
        except TypeError:
            assert fpart == ppart

    os.system("rm -r .testcache")
