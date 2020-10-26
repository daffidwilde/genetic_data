""" Tests for the crossover operator. """

import numpy as np
import pandas as pd
from hypothesis import settings

from edo import Family
from edo.distributions import Gamma, Normal, Poisson
from edo.individual import Individual, create_individual
from edo.operators import crossover

from .util.parameters import (
    INTEGER_CROSSOVER,
    INTEGER_TUPLE_CROSSOVER,
    TUPLE_CROSSOVER,
    TUPLE_INTEGER_CROSSOVER,
)


def _common_asserts(individual, *parents):
    """ Assert statements shared by multiple tests. """

    dataframe, metadata = individual

    assert isinstance(individual, Individual)
    assert isinstance(dataframe, pd.DataFrame)
    assert isinstance(metadata, list)
    assert isinstance(individual.random_state, np.random.RandomState)
    assert len(metadata) == len(dataframe.columns)

    for pdf in metadata:
        assert pdf in parents[0].metadata + parents[1].metadata

    for dtype, pdf in zip(dataframe.dtypes, metadata):
        assert dtype == pdf.dtype

    for i in [0, 1]:
        assert individual.dataframe.shape[i] in (
            parents[0].dataframe.shape[i],
            parents[1].dataframe.shape[i],
        )


@INTEGER_CROSSOVER
@settings(deadline=None)
def test_integer_limits(row_limits, col_limits, weights, prob, seed):
    """Verify that `crossover` produces a valid individual with all integer
    column limits."""

    distributions = [Gamma, Normal, Poisson]
    families = [Family(distribution) for distribution in distributions]
    random_state = np.random.RandomState(seed)

    parents = [
        create_individual(
            row_limits, col_limits, families, weights, random_state
        )
        for _ in [0, 1]
    ]

    individual = crossover(*parents, col_limits, families, random_state, prob)

    _common_asserts(individual, *parents)


@INTEGER_TUPLE_CROSSOVER
def test_integer_tuple_limits(row_limits, col_limits, weights, prob, seed):
    """Verify that `crossover` produces a valid individual where the lower and
    upper column limits are integer and tuple respectively."""

    distributions = [Gamma, Normal, Poisson]
    families = [Family(distribution) for distribution in distributions]
    random_state = np.random.RandomState(seed)

    parents = [
        create_individual(
            row_limits, col_limits, families, weights, random_state
        )
        for _ in [0, 1]
    ]

    individual = crossover(*parents, col_limits, families, random_state, prob)

    _common_asserts(individual, *parents)

    for family, upper_limit in zip(families, col_limits[1]):
        count = sum(pdf.family is family for pdf in individual.metadata)
        assert count <= upper_limit


@TUPLE_INTEGER_CROSSOVER
def test_tuple_integer_limits(row_limits, col_limits, weights, prob, seed):
    """Verify that `crossover` produces a valid individual where the lower and
    upper column limits are tuple and integer respectively."""

    distributions = [Gamma, Normal, Poisson]
    families = [Family(distribution) for distribution in distributions]
    random_state = np.random.RandomState(seed)

    parents = [
        create_individual(
            row_limits, col_limits, families, weights, random_state
        )
        for _ in [0, 1]
    ]

    individual = crossover(*parents, col_limits, families, random_state, prob)

    _common_asserts(individual, *parents)

    for family, lower_limit in zip(families, col_limits[0]):
        count = sum(pdf.family is family for pdf in individual.metadata)
        assert count >= lower_limit


@TUPLE_CROSSOVER
def test_tuple_limits(row_limits, col_limits, weights, prob, seed):
    """Verify that `crossover` produces a valid individual with all tuple
    column limits."""

    distributions = [Gamma, Normal, Poisson]
    families = [Family(distribution) for distribution in distributions]
    random_state = np.random.RandomState(seed)

    parents = [
        create_individual(
            row_limits, col_limits, families, weights, random_state
        )
        for _ in [0, 1]
    ]

    individual = crossover(*parents, col_limits, families, random_state, prob)

    _common_asserts(individual, *parents)
    for i, family in enumerate(families):
        count = sum(pdf.family is family for pdf in individual.metadata)
        assert col_limits[0][i] <= count <= col_limits[1][i]
