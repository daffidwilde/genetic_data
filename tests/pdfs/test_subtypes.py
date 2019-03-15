""" Tests for the generation of pdf subtypes. """

from hypothesis import given
from hypothesis.strategies import sampled_from

from edo.pdfs import all_pdfs
from edo.pdfs.subtypes import build_class


@given(cls=sampled_from(all_pdfs))
def test_make_subtype(cls):
    """ Test that a subtype can be made from a pdf class. """

    subtype = build_class(cls)
    for key, value in vars(subtype).items():
        if key in vars(cls) and key != "subtypes":
            assert getattr(cls, key) == value

    sub = subtype()
    assert isinstance(sub, subtype)
    assert not isinstance(sub, cls)
