""" Tests for the creation of an individual. """

import pandas as pd

from genetic_data.individual import Individual, create_individual
from genetic_data.pdfs import Gamma, Normal, Poisson

from .util.parameters import (
    INTEGER_INDIVIDUAL,
    INTEGER_TUPLE_INDIVIDUAL,
    TUPLE_INTEGER_INDIVIDUAL,
    TUPLE_INDIVIDUAL,
)


@INTEGER_INDIVIDUAL
def test_create_individual_int_int_lims(row_limits, col_limits, weights):
    """ Create an individual with all-integer column limits and verify that it
    is a namedtuple with a `pandas.DataFrame` field of a valid shape, and
    metadata made up of instances from the classes in pdfs. """

    pdfs = [Gamma, Normal, Poisson]
    individual = create_individual(row_limits, col_limits, pdfs, weights)
    dataframe, metadata = individual

    assert isinstance(individual, Individual)
    assert isinstance(metadata, list)
    assert isinstance(dataframe, pd.DataFrame)
    assert len(metadata) == len(dataframe.columns)

    for pdf in metadata:
        assert isinstance(pdf, tuple(pdfs))

    for i, limits in enumerate([row_limits, col_limits]):
        assert (
            dataframe.shape[i] >= limits[0] and dataframe.shape[i] <= limits[1]
        )


@INTEGER_TUPLE_INDIVIDUAL
def test_create_individual_int_tup_lims(row_limits, col_limits, weights):
    """ Create an individual with integer lower limits and tuple upper limits on
    the columns. Verify the individual is valid and of a reasonable shape and
    does not exceed the upper bounds. """

    pdfs = [Gamma, Normal, Poisson]
    individual = create_individual(row_limits, col_limits, pdfs, weights)
    dataframe, metadata = individual

    assert isinstance(individual, Individual)
    assert isinstance(metadata, list)
    assert isinstance(dataframe, pd.DataFrame)
    assert len(metadata) == len(dataframe.columns)

    for pdf in metadata:
        assert isinstance(pdf, tuple(pdfs))

    assert (
        dataframe.shape[0] >= row_limits[0]
        and dataframe.shape[0] <= row_limits[1]
    )
    assert dataframe.shape[1] >= col_limits[0] and dataframe.shape[1] <= sum(
        col_limits[1]
    )

    counts = {}
    for pdf_class in pdfs:
        counts[pdf_class] = 0
        for pdf in metadata:
            if isinstance(pdf, pdf_class):
                counts[pdf_class] += 1

    for i, count in enumerate(counts.values()):
        assert count <= col_limits[1][i]


@TUPLE_INTEGER_INDIVIDUAL
def test_create_individual_tup_int_lims(row_limits, col_limits, weights):
    """ Create an individual with tuple lower limits and integer upper limits on
    the columns. Verify the individual is valid and of a reasonable shape and
    does not exceed the lower bounds. """

    pdfs = [Gamma, Normal, Poisson]
    individual = create_individual(row_limits, col_limits, pdfs, weights)
    dataframe, metadata = individual

    assert isinstance(individual, Individual)
    assert isinstance(metadata, list)
    assert isinstance(dataframe, pd.DataFrame)
    assert len(metadata) == len(dataframe.columns)

    for pdf in metadata:
        assert isinstance(pdf, tuple(pdfs))

    assert (
        dataframe.shape[0] >= row_limits[0]
        and dataframe.shape[0] <= row_limits[1]
    )
    assert (
        dataframe.shape[1] >= sum(col_limits[0])
        and dataframe.shape[1] <= col_limits[1]
    )

    counts = {}
    for pdf_class in pdfs:
        counts[pdf_class] = 0
        for pdf in metadata:
            if isinstance(pdf, pdf_class):
                counts[pdf_class] += 1

    for i, count in enumerate(counts.values()):
        assert count >= col_limits[0][i]


@TUPLE_INDIVIDUAL
def test_create_individual_tup_tup_lims(row_limits, col_limits, weights):
    """ Create an individual with tuple column limits. Verify the individual is
    valid and of a reasonable shape and does not exceed either of the column
    bounds. """

    pdfs = [Gamma, Normal, Poisson]
    individual = create_individual(row_limits, col_limits, pdfs, weights)
    dataframe, metadata = individual

    assert isinstance(individual, Individual)
    assert isinstance(metadata, list)
    assert isinstance(dataframe, pd.DataFrame)
    assert len(metadata) == len(dataframe.columns)

    for pdf in metadata:
        assert isinstance(pdf, tuple(pdfs))

    assert (
        dataframe.shape[0] >= row_limits[0]
        and dataframe.shape[0] <= row_limits[1]
    )
    assert dataframe.shape[1] >= sum(col_limits[0]) and dataframe.shape[
        1
    ] <= sum(col_limits[1])

    counts = {}
    for pdf_class in pdfs:
        counts[pdf_class] = 0
        for pdf in metadata:
            if isinstance(pdf, pdf_class):
                counts[pdf_class] += 1

    for i, count in enumerate(counts.values()):
        assert count >= col_limits[0][i] and count <= col_limits[1][i]
