""" Tests for the calculating and writing of population fitness. """

import os
from pathlib import Path

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis.strategies import integers

import edo
from edo.distributions import Normal, Poisson, Uniform
from edo.fitness import get_fitness, get_population_fitness, write_fitness
from edo.individual import create_individual
from edo.population import create_initial_population

from .util.parameters import INTEGER_INDIVIDUAL, POP_FITNESS, POPULATION
from .util.trivials import trivial_fitness


@INTEGER_INDIVIDUAL
def test_get_fitness(row_limits, col_limits, weights, seed):
    """ Create an individual and get its fitness. Then verify that the fitness
    is of the correct data type and has been added to the cache. """

    distributions = [Normal, Poisson, Uniform]
    families = [edo.Family(dist) for dist in distributions]
    random_state = np.random.RandomState(seed)

    individual = create_individual(
        row_limits, col_limits, families, weights, random_state
    )

    fit = get_fitness(individual, trivial_fitness).compute()
    assert isinstance(fit, float)
    assert individual.fitness == fit


@INTEGER_INDIVIDUAL
def test_get_fitness_kwargs(row_limits, col_limits, weights, seed):
    """ Create an individual and get its fitness with keyword arguments. Then
    verify that the fitness is of the correct data type and has been added to
    the cache. """

    fitness_kwargs = {"arg": None}
    distributions = [Normal, Poisson, Uniform]
    families = [edo.Family(dist) for dist in distributions]
    random_state = np.random.RandomState(seed)

    individual = create_individual(
        row_limits, col_limits, families, weights, random_state
    )

    fit = get_fitness(individual, trivial_fitness, **fitness_kwargs).compute()
    assert isinstance(fit, float)
    assert individual.fitness == fit


@POPULATION
@settings(max_examples=30)
def test_get_population_fitness_serial(size, row_limits, col_limits, weights):
    """ Create a population and find its fitness serially. Verify that the
    fitness array is of the correct data type and size, and that they have each
    been added to the cache. """

    distributions = [Normal, Poisson, Uniform]
    families = [edo.Family(dist) for dist in distributions]
    random_states = {i: np.random.RandomState(i) for i in range(size)}

    population = create_initial_population(
        row_limits, col_limits, families, weights, random_states
    )

    pop_fit = get_population_fitness(population, trivial_fitness)
    assert len(pop_fit) == size
    for ind, fit in zip(population, pop_fit):
        assert isinstance(fit, float)
        assert ind.fitness == fit


@POP_FITNESS
@settings(max_examples=30)
def test_get_population_fitness_parallel(
    size, row_limits, col_limits, weights, processes
):
    """ Create a population and find its fitness in parallel. Verify that the
    fitness array is of the correct data type and size, and that they have each
    been added to the cache. """

    distributions = [Normal, Poisson, Uniform]
    families = [edo.Family(dist) for dist in distributions]
    random_states = {i: np.random.RandomState(i) for i in range(size)}

    population = create_initial_population(
        row_limits, col_limits, families, weights, random_states
    )

    pop_fit = get_population_fitness(population, trivial_fitness, processes)
    assert len(pop_fit) == size
    for ind, fit in zip(population, pop_fit):
        assert isinstance(fit, float)
        assert ind.fitness == fit


@given(size=integers(min_value=1, max_value=100))
def test_write_fitness(size):
    """ Test that a generation's fitness can be written to file correctly. """

    fitness = [trivial_fitness(pd.DataFrame()) for _ in range(size)]
    path = Path(".testcache")

    write_fitness(fitness, generation=0, root=path)
    write_fitness(fitness, generation=1, root=path)
    assert (path / "fitness.csv").exists()

    fit = pd.read_csv(path / "fitness.csv")
    assert list(fit.columns) == ["fitness", "generation", "individual"]
    assert list(fit.dtypes) == [float, int, int]
    assert list(fit["generation"].unique()) == [0, 1]
    assert list(fit["individual"]) == list(range(size)) * 2
    assert np.allclose(fit["fitness"].values, fitness * 2)

    os.system("rm -r .testcache")
