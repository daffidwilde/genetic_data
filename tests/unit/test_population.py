""" Tests for the creation of populations. """

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis.strategies import integers

from edo.fitness import get_fitness
from edo.individual import Individual
from edo.operators import selection
from edo.pdfs import Gamma, Normal, Poisson
from edo.population import create_initial_population, create_new_population

from .util.parameters import OFFSPRING, POPULATION
from .util.trivials import trivial_fitness


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

    bools = []
    for parent in parents:
        for ind in population:
            try:
                bools.append(np.all(parent.dataframe == ind.dataframe))
            except ValueError:
                bools.append(None)

    assert np.any(bools)

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
