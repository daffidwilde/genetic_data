""" Tests for the calculating and writing of population fitness. """

import os
from pathlib import Path

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis.strategies import integers

import edo
from edo.families import Normal, Poisson, Uniform
from edo.fitness import get_fitness, get_population_fitness, write_fitness
from edo.individual import create_individual
from edo.population import create_initial_population

from .util.parameters import INTEGER_INDIVIDUAL, POP_FITNESS, POPULATION
from .util.trivials import trivial_fitness


@INTEGER_INDIVIDUAL
def test_get_fitness(row_limits, col_limits, weights):
    """ Create an individual and get its fitness. Then verify that the fitness
    is of the correct data type and has been added to the cache. """

    cache = edo.cache

    families = [Normal, Poisson, Uniform]
    individual = create_individual(row_limits, col_limits, families, weights)
    dataframe = individual.dataframe

    fit = get_fitness(dataframe, trivial_fitness).compute()
    assert repr(dataframe) in cache
    assert isinstance(fit, float)

    cache.clear()


@INTEGER_INDIVIDUAL
def test_get_fitness_kwargs(row_limits, col_limits, weights):
    """ Create an individual and get its fitness with keyword arguments. Then
    verify that the fitness is of the correct data type and has been added to
    the cache. """

    cache = edo.cache

    fitness_kwargs = {"arg": None}
    families = [Normal, Poisson, Uniform]
    individual = create_individual(row_limits, col_limits, families, weights)
    dataframe = individual.dataframe

    fit = get_fitness(dataframe, trivial_fitness, fitness_kwargs).compute()
    assert repr(dataframe) in cache
    assert isinstance(fit, float)

    cache.clear()


@POPULATION
@settings(max_examples=30)
def test_get_population_fitness_serial(size, row_limits, col_limits, weights):
    """ Create a population and find its fitness serially. Verify that the
    fitness array is of the correct data type and size, and that they have each
    been added to the cache. """

    cache = edo.cache

    families = [Normal, Poisson, Uniform]
    population = create_initial_population(
        size, row_limits, col_limits, families, weights
    )

    pop_fit = get_population_fitness(population, trivial_fitness)
    assert len(pop_fit) == size
    for ind, fit in zip(population, pop_fit):
        assert repr(ind.dataframe) in cache
        assert isinstance(fit, float)

    cache.clear()


@POP_FITNESS
@settings(max_examples=30)
def test_get_population_fitness_parallel(
    size, row_limits, col_limits, weights, processes
):
    """ Create a population and find its fitness in parallel. Verify that the
    fitness array is of the correct data type and size, and that they have each
    been added to the cache. """

    cache = edo.cache

    families = [Normal, Poisson, Uniform]
    population = create_initial_population(
        size, row_limits, col_limits, families, weights
    )

    pop_fit = get_population_fitness(population, trivial_fitness, processes)
    assert len(pop_fit) == size
    for ind, fit in zip(population, pop_fit):
        assert repr(ind.dataframe) in cache
        assert isinstance(fit, float)

    cache.clear()


@given(size=integers(min_value=1, max_value=100))
def test_write_fitness(size):
    """ Test that a generation's fitness can be written to file correctly. """

    fitness = [trivial_fitness(pd.DataFrame()) for _ in range(size)]

    write_fitness(fitness, generation=0, root="out").compute()
    write_fitness(fitness, generation=1, root="out").compute()
    path = Path("out")
    assert (path / "fitness.csv").exists()

    fit = pd.read_csv(path / "fitness.csv")
    assert list(fit.columns) == ["fitness", "generation", "individual"]
    assert list(fit.dtypes) == [float, int, int]
    assert list(fit["generation"].unique()) == [0, 1]
    assert list(fit["individual"]) == list(range(size)) * 2
    assert np.allclose(fit["fitness"].values, fitness * 2)

    os.system("rm -r out")