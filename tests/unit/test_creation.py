""" Tests for the initial creation process. """

import numpy as np
import pandas as pd
import pytest

from hypothesis import given, settings
from hypothesis.strategies import integers

from genetic_data.creation import (
    create_individual,
    create_initial_population,
    create_new_population,
)
from genetic_data.operators import get_fitness, selection
from genetic_data.individual import Individual
from genetic_data.pdfs import Gamma, Normal, Poisson

from test_util.parameters import (
    INTEGER_INDIVIDUAL,
    INTEGER_TUPLE_INDIVIDUAL,
    TUPLE_INTEGER_INDIVIDUAL,
    TUPLE_INDIVIDUAL,
    POPULATION,
    OFFSPRING,
)
from test_util.trivials import trivial_fitness


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


@POPULATION
def test_create_initial_population(size, row_limits, col_limits, weights):
    """ Create an initial population of individuals and verify it is a list
    of the correct length with valid individuals. """

    pdfs = [Gamma, Normal, Poisson]
    population = create_initial_population(
        size, row_limits, col_limits, pdfs, weights
    )

    assert isinstance(population, list)
    assert len(population) == size

    for individual in population:
        dataframe, metadata = individual

        assert isinstance(individual, Individual)
        assert isinstance(metadata, list)
        assert isinstance(dataframe, pd.DataFrame)
        assert len(metadata) == len(dataframe.columns)

        for pdf in metadata:
            assert isinstance(pdf, tuple(pdfs))

        for i, limits in enumerate([row_limits, col_limits]):
            assert (
                dataframe.shape[i] >= limits[0]
                and dataframe.shape[i] <= limits[1]
            )


@given(size=integers(max_value=1))
def test_too_small_population(size):
    """ Verify that a `ValueError` is raised for small population sizes. """

    with pytest.raises(ValueError):
        create_initial_population(size, None, None, None, None)


@OFFSPRING
@settings(max_examples=25, deadline=None)
def test_create_new_population(
    size,
    row_limits,
    col_limits,
    weights,
    props,
    crossover_prob,
    mutation_prob,
    maximise,
):
    """ Create a population and use them to create a new proto-population
    of offspring. Verify that each offspring is a valid individual and there are
    the correct number of them. """

    best_prop, lucky_prop = props
    pdfs = [Gamma, Normal, Poisson]
    population = create_initial_population(
        size, row_limits, col_limits, pdfs, weights
    )
    pop_fitness = get_fitness(trivial_fitness, population)
    parents = selection(
        population, pop_fitness, best_prop, lucky_prop, maximise
    )

    population = create_new_population(
        parents,
        size,
        crossover_prob,
        mutation_prob,
        row_limits,
        col_limits,
        pdfs,
        weights,
    )

    assert isinstance(population, list)
    assert len(population) == size

    for parent in parents:
        try:
            assert np.any(
                [
                    np.all(parent.dataframe == ind.dataframe)
                    for ind in population
                ]
            )
        except:
            ValueError

    for individual in population:
        dataframe, metadata = individual

        assert isinstance(individual, Individual)
        assert isinstance(metadata, list)
        assert isinstance(dataframe, pd.DataFrame)
        assert len(metadata) == len(dataframe.columns)

        for pdf in metadata:
            assert isinstance(pdf, tuple(pdfs))

        for i, limits in enumerate([row_limits, col_limits]):
            assert (
                dataframe.shape[i] >= limits[0]
                and dataframe.shape[i] <= limits[1]
            )
