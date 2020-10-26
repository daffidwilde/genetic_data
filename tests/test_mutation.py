""" Tests for the mutation operator. """

import numpy as np
import pandas as pd

from edo import Family
from edo.distributions import Gamma, Normal, Poisson
from edo.individual import Individual, create_individual
from edo.operators import mutation

from .util.parameters import (
    INTEGER_MUTATION,
    INTEGER_TUPLE_MUTATION,
    TUPLE_INTEGER_MUTATION,
    TUPLE_MUTATION,
)


def _common_asserts(mutant, families):

    dataframe, metadata = mutant
    assert isinstance(mutant, Individual)
    assert isinstance(metadata, list)
    assert len(metadata) == len(dataframe.columns)
    assert isinstance(dataframe, pd.DataFrame)
    assert isinstance(mutant.random_state, np.random.RandomState)

    for pdf in metadata:
        assert sum(pdf.family is family for family in families) == 1


@INTEGER_MUTATION
def test_integer_limits(row_limits, col_limits, weights, prob, seed):
    """Verify that `mutation` creates a valid individual with all integer
    column limits."""

    distributions = [Gamma, Normal, Poisson]
    families = [Family(distribution) for distribution in distributions]
    state = np.random.RandomState(seed)

    individual = create_individual(
        row_limits, col_limits, families, weights, state
    )

    mutant = mutation(
        individual, prob, row_limits, col_limits, families, weights
    )

    _common_asserts(mutant, families)

    for i, limits in enumerate([row_limits, col_limits]):
        assert limits[0] <= mutant.dataframe.shape[i] <= limits[1]


@INTEGER_TUPLE_MUTATION
def test_integer_tuple_limits(row_limits, col_limits, weights, prob, seed):
    """Verify that `mutation` creates a valid individual where the lower and
    upper column limits are integer and tuple respectively."""

    distributions = [Gamma, Normal, Poisson]
    families = [Family(distribution) for distribution in distributions]
    state = np.random.RandomState(seed)

    individual = create_individual(
        row_limits, col_limits, families, weights, state
    )

    mutant = mutation(
        individual, prob, row_limits, col_limits, families, weights
    )

    _common_asserts(mutant, families)

    dataframe, metadata = mutant
    assert row_limits[0] <= dataframe.shape[0] <= row_limits[1]
    assert col_limits[0] <= dataframe.shape[1] <= sum(col_limits[1])

    family_counts = {
        family: sum(pdf.family is family for pdf in metadata)
        for family in families
    }

    for i, count in enumerate(family_counts.values()):
        assert count <= col_limits[1][i]


@TUPLE_INTEGER_MUTATION
def test_tuple_integer_limits(row_limits, col_limits, weights, prob, seed):
    """Verify that `mutation` creates a valid individual where the lower and
    upper column limits and tuple and integer respectively."""

    distributions = [Gamma, Normal, Poisson]
    families = [Family(distribution) for distribution in distributions]
    state = np.random.RandomState(seed)

    individual = create_individual(
        row_limits, col_limits, families, weights, state
    )

    mutant = mutation(
        individual, prob, row_limits, col_limits, families, weights
    )

    _common_asserts(mutant, families)

    dataframe, metadata = mutant
    assert row_limits[0] <= dataframe.shape[0] <= row_limits[1]
    assert sum(col_limits[0]) <= dataframe.shape[1] <= col_limits[1]

    family_counts = {
        family: sum(pdf.family is family for pdf in metadata)
        for family in families
    }

    for i, count in enumerate(family_counts.values()):
        assert count >= col_limits[0][i]


@TUPLE_MUTATION
def test_tuple_limits(row_limits, col_limits, weights, prob, seed):
    """Verify that `mutation` creates a valid individual with all tuple column
    limits."""

    distributions = [Gamma, Normal, Poisson]
    families = [Family(distribution) for distribution in distributions]
    state = np.random.RandomState(seed)

    individual = create_individual(
        row_limits, col_limits, families, weights, state
    )

    mutant = mutation(
        individual, prob, row_limits, col_limits, families, weights
    )

    _common_asserts(mutant, families)

    dataframe, metadata = mutant
    assert row_limits[0] <= dataframe.shape[0] <= row_limits[1]
    assert sum(col_limits[0]) <= dataframe.shape[1] <= sum(col_limits[1])

    family_counts = {
        family: sum(pdf.family is family for pdf in metadata)
        for family in families
    }

    for i, count in enumerate(family_counts.values()):
        assert col_limits[0][i] <= count <= col_limits[1][i]
