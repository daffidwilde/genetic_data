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
def test_int_int_lims(row_limits, col_limits, weights, prob):
    """ Verify that `crossover` produces a valid individual with all integer
    column limits. """

    pdfs = [Gamma, Normal, Poisson]
    parent1, parent2 = [
        create_individual(row_limits, col_limits, pdfs, weights)
        for _ in range(2)
    ]

    individual = crossover(parent1, parent2, col_limits, pdfs, prob)
    dataframe, metadata = individual

    assert isinstance(individual, Individual)
    assert isinstance(metadata, list)
    assert len(metadata) == len(dataframe.columns)
    assert isinstance(dataframe, pd.DataFrame)

    for pdf in metadata:
        assert pdf in parent1.metadata + parent2.metadata

    for dtype, pdf in zip(dataframe.dtypes, metadata):
        assert dtype == pdf.dtype

    for i in range(2):
        assert dataframe.shape[i] in [
            parent1.dataframe.shape[i],
            parent2.dataframe.shape[i],
        ]


@INTEGER_TUPLE_CROSSOVER
def test_int_tup_lims(row_limits, col_limits, weights, prob):
    """ Verify that `crossover` produces a valid individual where the lower and
    upper column limits are integer and tuple respectively. """

    pdfs = [Gamma, Normal, Poisson]
    parent1, parent2 = [
        create_individual(row_limits, col_limits, pdfs, weights)
        for _ in range(2)
    ]

    individual = crossover(parent1, parent2, col_limits, pdfs, prob)
    dataframe, metadata = individual

    assert isinstance(individual, Individual)
    assert isinstance(metadata, list)
    assert len(metadata) == len(dataframe.columns)
    assert isinstance(dataframe, pd.DataFrame)

    for pdf in metadata:
        assert pdf in parent1.metadata + parent2.metadata

    for dtype, pdf in zip(dataframe.dtypes, metadata):
        assert dtype == pdf.dtype

    for i in range(2):
        assert dataframe.shape[i] in [
            parent1.dataframe.shape[i],
            parent2.dataframe.shape[i],
        ]

    pdf_counts = {
        pdf_class: sum([isinstance(pdf, pdf_class) for pdf in metadata])
        for pdf_class in pdfs
    }

    for i, count in enumerate(pdf_counts.values()):
        assert count <= col_limits[1][i]


@TUPLE_INTEGER_CROSSOVER
def test_tup_int_lims(row_limits, col_limits, weights, prob):
    """ Verify that `crossover` produces a valid individual where the lower and
    upper column limits are tuple and integer respectively. """

    pdfs = [Gamma, Normal, Poisson]
    parent1, parent2 = [
        create_individual(row_limits, col_limits, pdfs, weights)
        for _ in range(2)
    ]

    individual = crossover(parent1, parent2, col_limits, pdfs, prob)
    dataframe, metadata = individual

    assert isinstance(individual, Individual)
    assert isinstance(metadata, list)
    assert len(metadata) == len(dataframe.columns)
    assert isinstance(dataframe, pd.DataFrame)

    for pdf in metadata:
        assert pdf in parent1.metadata + parent2.metadata

    for dtype, pdf in zip(dataframe.dtypes, metadata):
        assert dtype == pdf.dtype

    for i in range(2):
        assert dataframe.shape[i] in [
            parent1.dataframe.shape[i],
            parent2.dataframe.shape[i],
        ]

    pdf_counts = {
        pdf_class: sum([isinstance(pdf, pdf_class) for pdf in metadata])
        for pdf_class in pdfs
    }

    for i, count in enumerate(pdf_counts.values()):
        assert count >= col_limits[0][i]


@TUPLE_CROSSOVER
def test_tup_tup_lims(row_limits, col_limits, weights, prob):
    """ Verify that `crossover` produces a valid individual with all tuple
    column limits. """

    pdfs = [Gamma, Normal, Poisson]
    parent1, parent2 = [
        create_individual(row_limits, col_limits, pdfs, weights)
        for _ in range(2)
    ]

    individual = crossover(parent1, parent2, col_limits, pdfs, prob)
    dataframe, metadata = individual

    assert isinstance(individual, Individual)
    assert isinstance(metadata, list)
    assert len(metadata) == len(dataframe.columns)
    assert isinstance(dataframe, pd.DataFrame)

    for pdf in metadata:
        assert pdf in parent1.metadata + parent2.metadata

    for dtype, pdf in zip(dataframe.dtypes, metadata):
        assert dtype == pdf.dtype

    for i in range(2):
        assert dataframe.shape[i] in [
            parent1.dataframe.shape[i],
            parent2.dataframe.shape[i],
        ]

    pdf_counts = {
        pdf_class: sum([isinstance(pdf, pdf_class) for pdf in metadata])
        for pdf_class in pdfs
    }

    for i, count in enumerate(pdf_counts.values()):
        assert col_limits[0][i] <= count <= col_limits[1][i]
