""" Tests for the crossover operator. """

import pandas as pd
from hypothesis import settings

from edo.individual import Individual, create_individual
from edo.operators import crossover
from edo.pdfs import Gamma, Normal, Poisson

from .util.parameters import (
    INTEGER_CROSSOVER,
    INTEGER_TUPLE_CROSSOVER,
    TUPLE_CROSSOVER,
    TUPLE_INTEGER_CROSSOVER,
)


@INTEGER_CROSSOVER
@settings(deadline=None)
def test_integer_limits(row_limits, col_limits, weights, prob):
    """ Verify that `crossover` produces a valid individual with all integer
    column limits. """

    families = [Gamma, Normal, Poisson]
    for family in families:
        family.reset()

    parent1, parent2 = [
        create_individual(row_limits, col_limits, families, weights)
        for _ in [0, 1]
    ]

    individual = crossover(parent1, parent2, col_limits, families, prob)
    dataframe, metadata = individual

    assert isinstance(individual, Individual)
    assert isinstance(metadata, list)
    assert len(metadata) == len(dataframe.columns)
    assert isinstance(dataframe, pd.DataFrame)

    for pdf in metadata:
        assert pdf in parent1.metadata + parent2.metadata

    for dtype, pdf in zip(dataframe.dtypes, metadata):
        assert dtype == pdf.dtype

    for i in [0, 1]:
        assert dataframe.shape[i] in [
            parent1.dataframe.shape[i],
            parent2.dataframe.shape[i],
        ]


@INTEGER_TUPLE_CROSSOVER
def test_integer_tuple_limits(row_limits, col_limits, weights, prob):
    """ Verify that `crossover` produces a valid individual where the lower and
    upper column limits are integer and tuple respectively. """

    families = [Gamma, Normal, Poisson]
    for family in families:
        family.reset()

    parent1, parent2 = [
        create_individual(row_limits, col_limits, families, weights)
        for _ in [0, 1]
    ]

    individual = crossover(parent1, parent2, col_limits, families, prob)
    dataframe, metadata = individual

    assert isinstance(individual, Individual)
    assert isinstance(metadata, list)
    assert len(metadata) == len(dataframe.columns)
    assert isinstance(dataframe, pd.DataFrame)

    for pdf in metadata:
        assert pdf in parent1.metadata + parent2.metadata

    for dtype, pdf in zip(dataframe.dtypes, metadata):
        assert dtype == pdf.dtype

    for i in [0, 1]:
        assert dataframe.shape[i] in [
            parent1.dataframe.shape[i],
            parent2.dataframe.shape[i],
        ]

    for family, upper_limit in zip(families, col_limits[1]):
        count = sum([pdf.name == family.name for pdf in metadata])
        assert count <= upper_limit


@TUPLE_INTEGER_CROSSOVER
def test_tuple_integer_limits(row_limits, col_limits, weights, prob):
    """ Verify that `crossover` produces a valid individual where the lower and
    upper column limits are tuple and integer respectively. """

    families = [Gamma, Normal, Poisson]
    for family in families:
        family.reset()

    parent1, parent2 = [
        create_individual(row_limits, col_limits, families, weights)
        for _ in [0, 1]
    ]

    individual = crossover(parent1, parent2, col_limits, families, prob)
    dataframe, metadata = individual

    assert isinstance(individual, Individual)
    assert isinstance(metadata, list)
    assert len(metadata) == len(dataframe.columns)
    assert isinstance(dataframe, pd.DataFrame)

    for pdf in metadata:
        assert pdf in parent1.metadata + parent2.metadata

    for dtype, pdf in zip(dataframe.dtypes, metadata):
        assert dtype == pdf.dtype

    for i in [0, 1]:
        assert dataframe.shape[i] in [
            parent1.dataframe.shape[i],
            parent2.dataframe.shape[i],
        ]

    for family, lower_limit in zip(families, col_limits[0]):
        count = sum([pdf.name == family.name for pdf in metadata])
        assert count >= lower_limit


@TUPLE_CROSSOVER
def test_tuple_limits(row_limits, col_limits, weights, prob):
    """ Verify that `crossover` produces a valid individual with all tuple
    column limits. """

    families = [Gamma, Normal, Poisson]
    for family in families:
        family.reset()

    parent1, parent2 = [
        create_individual(row_limits, col_limits, families, weights)
        for _ in [0, 1]
    ]

    individual = crossover(parent1, parent2, col_limits, families, prob)
    dataframe, metadata = individual

    assert isinstance(individual, Individual)
    assert isinstance(metadata, list)
    assert len(metadata) == len(dataframe.columns)
    assert isinstance(dataframe, pd.DataFrame)

    for pdf in metadata:
        assert pdf in parent1.metadata + parent2.metadata

    for dtype, pdf in zip(dataframe.dtypes, metadata):
        assert dtype == pdf.dtype

    for i in [0, 1]:
        assert dataframe.shape[i] in [
            parent1.dataframe.shape[i],
            parent2.dataframe.shape[i],
        ]

    for i, family in enumerate(families):
        count = sum([pdf.name == family.name for pdf in metadata])
        assert col_limits[0][i] <= count <= col_limits[1][i]
