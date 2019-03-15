""" Tests for the mutation operator. """

import pandas as pd

from edo.individual import Individual, create_individual
from edo.operators import mutation
from edo.pdfs import Gamma, Normal, Poisson

from .util.parameters import (
    INTEGER_MUTATION,
    INTEGER_TUPLE_MUTATION,
    TUPLE_INTEGER_MUTATION,
    TUPLE_MUTATION,
)


@INTEGER_MUTATION
def test_int_int_lims(row_limits, col_limits, weights, prob):
    """ Verify that `mutation` creates a valid individual with all integer
    column limits. """

    pdfs = [Gamma, Normal, Poisson]
    individual = create_individual(row_limits, col_limits, pdfs, weights)

    mutant = mutation(individual, prob, row_limits, col_limits, pdfs, weights)
    dataframe, metadata = mutant

    assert isinstance(mutant, Individual)
    assert isinstance(metadata, list)
    assert len(metadata) == len(dataframe.columns)
    assert isinstance(dataframe, pd.DataFrame)

    for pdf in metadata:
        assert isinstance(pdf, tuple(pdfs))

    for i, limits in enumerate([row_limits, col_limits]):
        assert limits[0] <= dataframe.shape[i] <= limits[1]


@INTEGER_TUPLE_MUTATION
def test_int_tup_lims(row_limits, col_limits, weights, prob):
    """ Verify that `mutation` creates a valid individual where the lower and
    upper column limits are integer and tuple respectively. """

    pdfs = [Gamma, Normal, Poisson]
    individual = create_individual(row_limits, col_limits, pdfs, weights)

    mutant = mutation(individual, prob, row_limits, col_limits, pdfs, weights)
    dataframe, metadata = mutant

    assert isinstance(mutant, Individual)
    assert isinstance(metadata, list)
    assert len(metadata) == len(dataframe.columns)
    assert isinstance(dataframe, pd.DataFrame)

    for pdf in metadata:
        assert isinstance(pdf, tuple(pdfs))

    assert row_limits[0] <= dataframe.shape[0] <= row_limits[1]
    assert col_limits[0] <= dataframe.shape[1] <= sum(col_limits[1])

    pdf_counts = {
        pdf_class: sum([isinstance(pdf, pdf_class) for pdf in metadata])
        for pdf_class in pdfs
    }

    for i, count in enumerate(pdf_counts.values()):
        assert count <= col_limits[1][i]


@TUPLE_INTEGER_MUTATION
def test_tup_int_lims(row_limits, col_limits, weights, prob):
    """ Verify that `mutation` creates a valid individual where the lower and
    upper column limits and tuple and integer respectively. """

    pdfs = [Gamma, Normal, Poisson]
    individual = create_individual(row_limits, col_limits, pdfs, weights)

    mutant = mutation(individual, prob, row_limits, col_limits, pdfs, weights)
    dataframe, metadata = mutant

    assert isinstance(mutant, Individual)
    assert isinstance(metadata, list)
    assert len(metadata) == len(dataframe.columns)
    assert isinstance(dataframe, pd.DataFrame)

    for pdf in metadata:
        assert isinstance(pdf, tuple(pdfs))

    assert row_limits[0] <= dataframe.shape[0] <= row_limits[1]
    assert sum(col_limits[0]) <= dataframe.shape[1] <= col_limits[1]

    pdf_counts = {
        pdf_class: sum([isinstance(pdf, pdf_class) for pdf in metadata])
        for pdf_class in pdfs
    }

    for i, count in enumerate(pdf_counts.values()):
        assert count >= col_limits[0][i]


@TUPLE_MUTATION
def test_tup_tup_lims(row_limits, col_limits, weights, prob):
    """ Verify that `mutation` creates a valid individual with all tuple column
    limits. """

    pdfs = [Gamma, Normal, Poisson]
    individual = create_individual(row_limits, col_limits, pdfs, weights)

    mutant = mutation(individual, prob, row_limits, col_limits, pdfs, weights)
    dataframe, metadata = mutant

    assert isinstance(mutant, Individual)
    assert isinstance(metadata, list)
    assert len(metadata) == len(dataframe.columns)
    assert isinstance(dataframe, pd.DataFrame)

    for pdf in metadata:
        assert isinstance(pdf, tuple(pdfs))

    assert row_limits[0] <= dataframe.shape[0] <= row_limits[1]
    assert sum(col_limits[0]) <= dataframe.shape[1] <= sum(col_limits[1])

    pdf_counts = {
        pdf_class: sum([isinstance(pdf, pdf_class) for pdf in metadata])
        for pdf_class in pdfs
    }

    for i, count in enumerate(pdf_counts.values()):
        assert col_limits[0][i] <= count <= col_limits[1][i]
