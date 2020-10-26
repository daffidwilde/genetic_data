""" Tests for the creation of an individual. """

import os
from pathlib import Path

import numpy as np
import pandas as pd
from hypothesis import given, settings
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


def _common_asserts(individual, state, families):
    """ Assert statements called in most tests. """

    dataframe, metadata = individual

    assert isinstance(individual, Individual)
    assert isinstance(dataframe, pd.DataFrame)
    assert isinstance(metadata, list)
    assert isinstance(individual.random_state, np.random.RandomState)

    assert individual.random_state is state
    assert len(metadata) == len(dataframe.columns)
    for pdf in metadata:
        assert sum(pdf.family is family for family in families) == 1


@given(dataframe=text(), metadata=text())
def test_repr(dataframe, metadata):
    """ Test an individual has the correct representation. """

    individual = Individual(dataframe, metadata)
    assert (
        repr(individual)
        == f"Individual(dataframe={dataframe}, metadata={metadata})"
    )


@INTEGER_INDIVIDUAL
def test_integer_limits(row_limits, col_limits, weights, seed):
    """Create an individual with all-integer column limits and verify that it
    is a namedtuple with a `pandas.DataFrame` field of a valid shape, and
    metadata made up of instances from the classes in families."""

    distributions = [Gamma, Normal, Poisson]
    families = [Family(distribution) for distribution in distributions]

    state = np.random.RandomState(seed)

    individual = create_individual(
        row_limits, col_limits, families, weights, state
    )

    _common_asserts(individual, state, families)

    for i, limits in enumerate([row_limits, col_limits]):
        assert limits[0] <= individual.dataframe.shape[i] <= limits[1]


@INTEGER_TUPLE_INDIVIDUAL
def test_integer_tuple_limits(row_limits, col_limits, weights, seed):
    """Create an individual with integer lower limits and tuple upper limits on
    the columns. Verify the individual is valid and of a reasonable shape and
    does not exceed the upper bounds."""

    distributions = [Gamma, Normal, Poisson]
    families = [Family(distribution) for distribution in distributions]

    state = np.random.RandomState(seed)

    individual = create_individual(
        row_limits, col_limits, families, weights, state
    )

    _common_asserts(individual, state, families)

    dataframe, metadata = individual
    assert row_limits[0] <= dataframe.shape[0] <= row_limits[1]
    assert col_limits[0] <= dataframe.shape[1] <= sum(col_limits[1])

    for family, upper_limit in zip(families, col_limits[1]):
        count = sum(isinstance(pdf, family.distribution) for pdf in metadata)
        assert count <= upper_limit


@TUPLE_INTEGER_INDIVIDUAL
def test_tuple_integer_limits(row_limits, col_limits, weights, seed):
    """Create an individual with tuple lower limits and integer upper limits on
    the columns. Verify the individual is valid and of a reasonable shape and
    does not exceed the lower bounds."""

    distributions = [Gamma, Normal, Poisson]
    families = [Family(distribution) for distribution in distributions]

    state = np.random.RandomState(seed)

    individual = create_individual(
        row_limits, col_limits, families, weights, state
    )

    _common_asserts(individual, state, families)

    dataframe, metadata = individual
    assert row_limits[0] <= dataframe.shape[0] <= row_limits[1]
    assert sum(col_limits[0]) <= dataframe.shape[1] <= col_limits[1]

    for family, lower_limit in zip(families, col_limits[0]):
        count = sum([isinstance(pdf, family.distribution) for pdf in metadata])
        assert count >= lower_limit


@TUPLE_INDIVIDUAL
def test_tuple_limits(row_limits, col_limits, weights, seed):
    """Create an individual with tuple column limits. Verify the individual is
    valid and of a reasonable shape and does not exceed either of the column
    bounds."""

    distributions = [Gamma, Normal, Poisson]
    families = [Family(distribution) for distribution in distributions]

    state = np.random.RandomState(seed)

    individual = create_individual(
        row_limits, col_limits, families, weights, state
    )

    _common_asserts(individual, state, families)

    dataframe, metadata = individual
    assert row_limits[0] <= dataframe.shape[0] <= row_limits[1]
    assert sum(col_limits[0]) <= dataframe.shape[1] <= sum(col_limits[1])

    for i, family in enumerate(families):
        count = sum([isinstance(pdf, family.distribution) for pdf in metadata])
        assert col_limits[0][i] <= count <= col_limits[1][i]


@INTEGER_INDIVIDUAL
@settings(deadline=None)
def test_to_and_from_file(row_limits, col_limits, weights, seed):
    """ Test that an individual can be saved to and created from file. """

    path = Path(".testcache/individual")

    distributions = [Gamma, Normal, Poisson]
    families = [Family(distribution) for distribution in distributions]

    state = np.random.RandomState(seed)

    individual = create_individual(
        row_limits, col_limits, families, weights, state
    )

    individual.to_file(path, ".testcache")
    assert (path / "main.csv").exists()
    assert (path / "main.meta").exists()
    assert (path / "main.state").exists()

    saved_individual = Individual.from_file(path, distributions, ".testcache")

    assert np.allclose(
        saved_individual.dataframe.values, individual.dataframe.values
    )

    for saved_pdf, pdf in zip(saved_individual.metadata, individual.metadata):

        assert saved_pdf.family.name == pdf.family.name
        assert saved_pdf.family.distribution is pdf.family.distribution
        assert saved_pdf.to_dict() == pdf.to_dict()

    for saved_part, state_part in zip(
        saved_individual.random_state.get_state(), state.get_state()
    ):
        try:
            assert all(saved_part == state_part)
        except TypeError:
            assert saved_part == state_part

    os.system("rm -r .testcache")
