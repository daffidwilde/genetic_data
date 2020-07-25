""" Tests for the creation of populations. """

import numpy as np
import pandas as pd
from hypothesis import settings

from edo import Family
from edo.distributions import Gamma, Normal, Poisson
from edo.individual import Individual
from edo.population import create_initial_population, create_new_population

from .util.parameters import OFFSPRING, POPULATION


@POPULATION
def test_create_initial_population(size, row_limits, col_limits, weights):
    """ Create an initial population of individuals and verify it is a list
    of the correct length with valid individuals. """

    distributions = [Gamma, Normal, Poisson]
    families = [Family(distribution) for distribution in distributions]
    states = {i: np.random.RandomState(i) for i in range(size)}

    population = create_initial_population(
        row_limits, col_limits, families, weights, states
    )

    assert isinstance(population, list)
    assert len(population) == size

    for individual in population:
        dataframe, metadata = individual

        assert isinstance(individual, Individual)
        assert isinstance(metadata, list)
        assert isinstance(dataframe, pd.DataFrame)
        assert isinstance(individual.random_state, np.random.RandomState)
        assert len(metadata) == len(dataframe.columns)

        for dtype, pdf in zip(dataframe.dtypes, metadata):
            assert sum(pdf.family is family for family in families) == 1
            assert pdf.dtype == dtype

        for i, limits in enumerate([row_limits, col_limits]):
            assert limits[0] <= dataframe.shape[i] <= limits[1]


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

    distributions = [Gamma, Normal, Poisson]
    families = [Family(distribution) for distribution in distributions]
    states = {i: np.random.RandomState(i) for i in range(size)}

    population = create_initial_population(
        row_limits, col_limits, families, weights, states
    )
    parent_size = max(int(size / 2), 2)
    parents = population[:parent_size]

    population = create_new_population(
        parents,
        population,
        crossover_prob,
        mutation_prob,
        row_limits,
        col_limits,
        families,
        weights,
        states,
    )

    assert isinstance(population, list)
    assert len(population) == size

    for i, parent in enumerate(parents):
        individual = population[i]
        assert parent.dataframe.equals(individual.dataframe)
        assert parent.metadata == individual.metadata
        assert parent.random_state == individual.random_state

    for individual in population:
        dataframe, metadata = individual

        assert isinstance(individual, Individual)
        assert isinstance(metadata, list)
        assert isinstance(dataframe, pd.DataFrame)
        assert isinstance(individual.random_state, np.random.RandomState)
        assert len(metadata) == len(dataframe.columns)

        for dtype, pdf in zip(dataframe.dtypes, metadata):
            assert sum(pdf.family is family for family in families) == 1
            assert dtype == pdf.dtype

        for i, limits in enumerate([row_limits, col_limits]):
            assert limits[0] <= dataframe.shape[i] <= limits[1]
