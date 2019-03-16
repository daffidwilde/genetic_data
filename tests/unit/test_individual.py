""" Tests for the creation of an individual. """

import pandas as pd

from edo.individual import Individual, create_individual
from edo.pdfs import Gamma, Normal, Poisson

from .util.parameters import (
    INTEGER_INDIVIDUAL,
    INTEGER_TUPLE_INDIVIDUAL,
    TUPLE_INDIVIDUAL,
    TUPLE_INTEGER_INDIVIDUAL,
)


@INTEGER_INDIVIDUAL
def test_integer_limits(row_limits, col_limits, weights):
    """ Create an individual with all-integer column limits and verify that it
    is a namedtuple with a `pandas.DataFrame` field of a valid shape, and
    metadata made up of instances from the classes in families. """

    families = [Gamma, Normal, Poisson]
    for family in families:
        family.reset()

    individual = create_individual(row_limits, col_limits, families, weights)
    dataframe, metadata = individual

    assert isinstance(individual, Individual)
    assert isinstance(metadata, list)
    assert isinstance(dataframe, pd.DataFrame)
    assert len(metadata) == len(dataframe.columns)

    for pdf in metadata:
        for family in families:
            if pdf.name == family.name:
                assert pdf.__class__ in family.subtypes

    for i, limits in enumerate([row_limits, col_limits]):
        assert limits[0] <= dataframe.shape[i] <= limits[1]


@INTEGER_TUPLE_INDIVIDUAL
def test_integer_tuple_limits(row_limits, col_limits, weights):
    """ Create an individual with integer lower limits and tuple upper limits on
    the columns. Verify the individual is valid and of a reasonable shape and
    does not exceed the upper bounds. """

    families = [Gamma, Normal, Poisson]
    individual = create_individual(row_limits, col_limits, families, weights)
    dataframe, metadata = individual

    assert isinstance(individual, Individual)
    assert isinstance(metadata, list)
    assert isinstance(dataframe, pd.DataFrame)
    assert len(metadata) == len(dataframe.columns)

    for pdf in metadata:
        for family in families:
            if pdf.name == family.name:
                assert pdf.__class__ in family.subtypes

    assert row_limits[0] <= dataframe.shape[0] <= row_limits[1]
    assert col_limits[0] <= dataframe.shape[1] <= sum(col_limits[1])

    for family, upper_limit in zip(families, col_limits[1]):
        count = sum([pdf.name == family.name for pdf in metadata])
        assert count <= upper_limit


@TUPLE_INTEGER_INDIVIDUAL
def test_tuple_integer_limits(row_limits, col_limits, weights):
    """ Create an individual with tuple lower limits and integer upper limits on
    the columns. Verify the individual is valid and of a reasonable shape and
    does not exceed the lower bounds. """

    families = [Gamma, Normal, Poisson]
    individual = create_individual(row_limits, col_limits, families, weights)
    dataframe, metadata = individual

    assert isinstance(individual, Individual)
    assert isinstance(metadata, list)
    assert isinstance(dataframe, pd.DataFrame)
    assert len(metadata) == len(dataframe.columns)

    for pdf in metadata:
        for family in families:
            if pdf.name == family.name:
                assert pdf.__class__ in family.subtypes

    assert row_limits[0] <= dataframe.shape[0] <= row_limits[1]
    assert sum(col_limits[0]) <= dataframe.shape[1] <= col_limits[1]

    for family, lower_limit in zip(families, col_limits[0]):
        count = sum([pdf.name == family.name for pdf in metadata])
        assert count >= lower_limit


@TUPLE_INDIVIDUAL
def test_tuple_limits(row_limits, col_limits, weights):
    """ Create an individual with tuple column limits. Verify the individual is
    valid and of a reasonable shape and does not exceed either of the column
    bounds. """

    families = [Gamma, Normal, Poisson]
    individual = create_individual(row_limits, col_limits, families, weights)
    dataframe, metadata = individual

    assert isinstance(individual, Individual)
    assert isinstance(metadata, list)
    assert isinstance(dataframe, pd.DataFrame)
    assert len(metadata) == len(dataframe.columns)

    for pdf in metadata:
        for family in families:
            if pdf.name == family.name:
                assert pdf.__class__ in family.subtypes

    assert row_limits[0] <= dataframe.shape[0] <= row_limits[1]
    assert sum(col_limits[0]) <= dataframe.shape[1] <= sum(col_limits[1])

    for i, family in enumerate(families):
        count = sum([pdf.name == family.name for pdf in metadata])
        assert col_limits[0][i] <= count <= col_limits[1][i]
