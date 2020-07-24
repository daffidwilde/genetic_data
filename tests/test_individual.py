""" Tests for the creation of an individual. """

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from hypothesis import given
from hypothesis.strategies import text

from edo import Family
from edo.distributions import Gamma, Normal, Poisson
from edo.individual import Individual, create_individual

from .util.parameters import (
    INTEGER_INDIVIDUAL,
    INTEGER_TUPLE_INDIVIDUAL,
    TUPLE_INDIVIDUAL,
    TUPLE_INTEGER_INDIVIDUAL,
)


@given(dataframe=text(), metadata=text())
def test_repr(dataframe, metadata):
    """ Test an individual has the correct representation. """

    individual = Individual(dataframe, metadata)
    assert (
        repr(individual)
        == f"Individual(dataframe={dataframe}, metadata={metadata})"
    )


@INTEGER_INDIVIDUAL
def test_integer_limits(row_limits, col_limits, weights):
    """ Create an individual with all-integer column limits and verify that it
    is a namedtuple with a `pandas.DataFrame` field of a valid shape, and
    metadata made up of instances from the classes in families. """

    distributions = [Gamma, Normal, Poisson]
    families = [Family(distribution) for distribution in distributions]

    individual = create_individual(
        row_limits, col_limits, families, weights
    )
    dataframe, metadata = individual

    assert isinstance(individual, Individual)
    assert isinstance(metadata, list)
    assert isinstance(dataframe, pd.DataFrame)
    assert len(metadata) == len(dataframe.columns)

    for pdf in metadata:
        for family in families:
            if isinstance(pdf, family.distribution):
                assert pdf.__class__ in family.subtypes.values()

    for i, limits in enumerate([row_limits, col_limits]):
        assert limits[0] <= dataframe.shape[i] <= limits[1]


@INTEGER_TUPLE_INDIVIDUAL
def test_integer_tuple_limits(row_limits, col_limits, weights):
    """ Create an individual with integer lower limits and tuple upper limits on
    the columns. Verify the individual is valid and of a reasonable shape and
    does not exceed the upper bounds. """

    distributions = [Gamma, Normal, Poisson]
    families = [Family(distribution) for distribution in distributions]

    individual = create_individual(
        row_limits, col_limits, families, weights
    )
    dataframe, metadata = individual

    assert isinstance(individual, Individual)
    assert isinstance(metadata, list)
    assert isinstance(dataframe, pd.DataFrame)
    assert len(metadata) == len(dataframe.columns)

    for pdf in metadata:
        for family in families:
            if isinstance(pdf, family.distribution):
                assert pdf.__class__ in family.subtypes.values()

    assert row_limits[0] <= dataframe.shape[0] <= row_limits[1]
    assert col_limits[0] <= dataframe.shape[1] <= sum(col_limits[1])

    for family, upper_limit in zip(families, col_limits[1]):
        count = sum([isinstance(pdf, family.distribution) for pdf in metadata])
        assert count <= upper_limit


@TUPLE_INTEGER_INDIVIDUAL
def test_tuple_integer_limits(row_limits, col_limits, weights):
    """ Create an individual with tuple lower limits and integer upper limits on
    the columns. Verify the individual is valid and of a reasonable shape and
    does not exceed the lower bounds. """

    distributions = [Gamma, Normal, Poisson]
    families = [Family(distribution) for distribution in distributions]

    individual = create_individual(
        row_limits, col_limits, families, weights
    )
    dataframe, metadata = individual

    assert isinstance(individual, Individual)
    assert isinstance(metadata, list)
    assert isinstance(dataframe, pd.DataFrame)
    assert len(metadata) == len(dataframe.columns)

    for pdf in metadata:
        for family in families:
            if isinstance(pdf, family.distribution):
                assert pdf.__class__ in family.subtypes.values()

    assert row_limits[0] <= dataframe.shape[0] <= row_limits[1]
    assert sum(col_limits[0]) <= dataframe.shape[1] <= col_limits[1]

    for family, lower_limit in zip(families, col_limits[0]):
        count = sum([isinstance(pdf, family.distribution) for pdf in metadata])
        assert count >= lower_limit


@TUPLE_INDIVIDUAL
def test_tuple_limits(row_limits, col_limits, weights):
    """ Create an individual with tuple column limits. Verify the individual is
    valid and of a reasonable shape and does not exceed either of the column
    bounds. """

    distributions = [Gamma, Normal, Poisson]
    families = [Family(distribution) for distribution in distributions]

    individual = create_individual(
        row_limits, col_limits, families, weights
    )
    dataframe, metadata = individual

    assert isinstance(individual, Individual)
    assert isinstance(metadata, list)
    assert isinstance(dataframe, pd.DataFrame)
    assert len(metadata) == len(dataframe.columns)

    for pdf in metadata:
        for family in families:
            if isinstance(pdf, family.distribution):
                assert pdf.__class__ in family.subtypes.values()

    assert row_limits[0] <= dataframe.shape[0] <= row_limits[1]
    assert sum(col_limits[0]) <= dataframe.shape[1] <= sum(col_limits[1])

    for i, family in enumerate(families):
        count = sum([isinstance(pdf, family.distribution) for pdf in metadata])
        assert col_limits[0][i] <= count <= col_limits[1][i]


@INTEGER_INDIVIDUAL
def test_to_and_from_file(row_limits, col_limits, weights):
    """ Test that an individual can be saved to and created from file. """

    path = Path(".testcache/individual")

    distributions = [Gamma, Normal, Poisson]
    families = [Family(distribution) for distribution in distributions]

    individual = create_individual(
        row_limits, col_limits, families, weights
    )

    individual.to_file(path, ".testcache")
    assert (path / "main.csv").exists()
    assert (path / "main.meta").exists()

    saved_individual = Individual.from_file(path, distributions, ".testcache")

    assert np.allclose(
        saved_individual.dataframe.values, individual.dataframe.values
    )

    for saved_pdf, pdf in zip(saved_individual.metadata, individual.metadata):

        assert saved_pdf.family.name == pdf.family.name
        assert saved_pdf.family.distribution is pdf.family.distribution
        assert saved_pdf.to_dict() == pdf.to_dict()

    os.system("rm -r .testcache")
