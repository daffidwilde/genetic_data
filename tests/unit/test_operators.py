""" Tests for the crossover and mutation operator scripts. """

import numpy as np
import pandas as pd

import pytest

from hypothesis import settings

from genetic_data.creation import create_individual, create_initial_population
from genetic_data.individual import Individual
from genetic_data.operators import crossover, get_fitness, mutation, selection
from genetic_data.pdfs import Gamma, Normal, Poisson

from test_util.parameters import (
    CROSSOVER,
    FITNESS,
    MUTATION,
    SELECTION,
    SMALL_PROPS,
)
from test_util.trivials import trivial_fitness


@FITNESS
def test_get_fitness(size, row_limits, col_limits, weights):
    """ Create a population and get its fitness. Then verify that the
    fitness is of the correct size and data type. """

    pdfs = [Gamma, Normal, Poisson]
    population = create_initial_population(
        size, row_limits, col_limits, pdfs, weights
    )
    pop_fitness = get_fitness(trivial_fitness, population)

    assert len(pop_fitness) == size
    assert np.array(pop_fitness).dtype == "float"


@SELECTION
def test_selection(size, row_limits, col_limits, weights, props, maximise):
    """ Create a population, get its fitness and select potential parents
    based on that fitness. Verify that parents are all valid individuals. """

    best_prop, lucky_prop = props
    pdfs = [Gamma, Normal, Poisson]
    population = create_initial_population(
        size, row_limits, col_limits, pdfs, weights
    )
    pop_fitness = get_fitness(trivial_fitness, population)
    parents = selection(
        population, pop_fitness, best_prop, lucky_prop, maximise
    )

    assert len(parents) == min(
        size, int(best_prop * size) + int(lucky_prop * size)
    )

    for individual in parents:
        metadata, dataframe = individual

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


@SMALL_PROPS
def test_selection_raises_error(
    size, row_limits, col_limits, weights, props, maximise
):
    """ Assert that best and lucky proportions must be sensible. """

    with pytest.raises(ValueError):
        best_prop, lucky_prop = props
        pdfs = [Gamma, Normal, Poisson]
        population = create_initial_population(
            size, row_limits, col_limits, pdfs, weights
        )
        pop_fitness = get_fitness(trivial_fitness, population)

        selection(population, pop_fitness, best_prop, lucky_prop, maximise)


@CROSSOVER
@settings(deadline=None)
def test_crossover(row_limits, col_limits, weights, prob):
    """ Verify that `crossover` produces a valid individual. """

    pdfs = [Gamma, Normal, Poisson]
    parent1, parent2 = [
        create_individual(row_limits, col_limits, pdfs, weights)
        for _ in range(2)
    ]

    individual = crossover(parent1, parent2, prob)
    metadata, dataframe = individual

    assert isinstance(individual, Individual)
    assert isinstance(metadata, list)
    assert len(metadata) == len(dataframe.columns)
    assert isinstance(dataframe, pd.DataFrame)

    for pdf in metadata:
        assert pdf in parent1.column_metadata + parent2.column_metadata

    for i in range(2):
        assert dataframe.shape[i] in [
            parent1.dataframe.shape[i],
            parent2.dataframe.shape[i],
        ]


@MUTATION
def test_mutation(row_limits, col_limits, weights, prob):
    """ Verify that `mutation` creates a valid individual. """

    pdfs = [Gamma, Normal, Poisson]
    individual = create_individual(row_limits, col_limits, pdfs, weights)
    mutant = mutation(individual, prob, row_limits, col_limits, pdfs, weights)

    metadata, dataframe = mutant

    assert isinstance(mutant, Individual)
    assert isinstance(metadata, list)
    assert len(metadata) == len(dataframe.columns)
    assert isinstance(dataframe, pd.DataFrame)

    for pdf in metadata:
        assert isinstance(pdf, tuple(pdfs))

    for axis in [0, 1]:
        assert (
            dataframe.shape[axis] >= individual.dataframe.shape[axis]
            or dataframe.shape[axis] <= individual.dataframe.shape[axis]
        )
