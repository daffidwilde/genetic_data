""" Tests for the helper functions in the running of the EDO algorithm. """

import os

import dask.dataframe as dd
import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis.strategies import integers

from edo.individual import Individual, create_individual
from edo.pdfs import Normal, Poisson, Uniform
from edo.population import create_initial_population
from edo.run import (
    _get_fit_history,
    _get_metadata_dicts,
    _get_pop_history,
    _initialise_algorithm,
    _update_fit_history,
    _update_pop_history,
    _update_subtypes,
)
from edo.write import write_generation

from .util.parameters import INTEGER_INDIVIDUAL, POPULATION
from .util.trivials import trivial_fitness


@POPULATION
def test_initialise_algorithm(size, row_limits, col_limits, weights):
    """ Test that the algorithm can be initialised correctly with a population
    and its fitness. """

    families = [Normal, Poisson, Uniform]
    fitness_kwargs = {"arg": None}
    population, pop_fitness = _initialise_algorithm(
        trivial_fitness,
        size,
        row_limits,
        col_limits,
        families,
        weights,
        None,
        fitness_kwargs,
    )

    assert isinstance(population, list)
    assert len(population) == len(pop_fitness) == size

    for individual, fitness in zip(population, pop_fitness):
        assert isinstance(individual, Individual)
        assert isinstance(fitness, float)


@INTEGER_INDIVIDUAL
def test_get_metadata_dicts(row_limits, col_limits, weights):
    """ Test that the metadata for an individual can be properly converted to a
    dictionary. """

    families = [Normal, Poisson, Uniform]
    individual = create_individual(row_limits, col_limits, families, weights)
    meta_dicts = _get_metadata_dicts(individual)

    assert isinstance(meta_dicts, list)

    for meta in meta_dicts:
        assert isinstance(meta, dict)
        assert meta["name"] in [family.name for family in families]


@POPULATION
def test_update_pop_history(size, row_limits, col_limits, weights):
    """ Test that the population history can be updated. """

    families = [Normal, Poisson, Uniform]
    population = create_initial_population(
        size, row_limits, col_limits, families, weights
    )
    pop_history = _update_pop_history(population)

    assert len(pop_history) == 1
    for i, individual in enumerate(pop_history[-1]):
        assert individual.dataframe.equals(population[i].dataframe)
        assert individual.metadata == [
            m.to_dict() for m in population[i].metadata
        ]

    pop_history = _update_pop_history(population, pop_history)

    assert len(pop_history) == 2
    for i, individual in enumerate(pop_history[-1]):
        assert individual.dataframe.equals(population[i].dataframe)
        assert individual.metadata == [
            m.to_dict() for m in population[i].metadata
        ]


@given(size=integers(min_value=1, max_value=100))
def test_update_fit_history(size):
    """ Test that the fitness history can be updated. """

    fitness = [0.5] * size
    fit_history = _update_fit_history(fitness, itr=0)

    assert isinstance(fit_history, pd.DataFrame)
    assert fit_history.shape == (size, 3)
    assert list(fit_history["fitness"].values) == fitness
    assert list(fit_history["generation"].unique()) == [0]
    assert list(fit_history["individual"]) == list(range(size))

    fit_history = _update_fit_history(fitness, itr=1, fit_history=fit_history)

    assert fit_history.shape == (size * 2, 3)
    assert list(fit_history["fitness"].values) == fitness * 2
    assert list(fit_history["generation"].unique()) == [0, 1]
    assert list(fit_history["individual"]) == list(range(size)) * 2


@POPULATION
def test_update_subtypes(size, row_limits, col_limits, weights):
    """ Test that the subtypes of the present distributions can be updated. """

    families = [Normal, Poisson, Uniform]
    population = create_initial_population(
        size, row_limits, col_limits, families, weights
    )

    parents = population[: max(int(size / 5), 1)]
    parent_subtypes = {
        pdf.__class__ for parent in parents for pdf in parent.metadata
    }

    families = _update_subtypes(parents, families)
    updated_subtypes = {sub for family in families for sub in family.subtypes}

    assert parent_subtypes == updated_subtypes


@settings(deadline=None, max_examples=10)
@POPULATION
def test_get_history(size, row_limits, col_limits, weights):
    """ Test that a population and fitness history can be read in correctly. """

    families = [Normal, Poisson, Uniform]
    population = create_initial_population(
        size, row_limits, col_limits, families, weights
    )
    pop_fitness = [
        trivial_fitness(individual.dataframe) for individual in population
    ]

    write_generation(population, pop_fitness, gen=0, root="out", processes=4)

    pop_history = _get_pop_history(root="out", itr=1)
    fit_history = _get_fit_history(root="out")

    assert isinstance(pop_history, list)
    for generation in pop_history:

        assert isinstance(generation, list)
        for i, individual in enumerate(generation):

            pop_ind = population[i]
            assert isinstance(individual.dataframe, dd.DataFrame)
            assert np.allclose(
                pop_ind.dataframe.values, individual.dataframe.values.compute()
            )
            assert isinstance(individual.metadata, list)
            assert individual.metadata == [
                m.to_dict() for m in pop_ind.metadata
            ]

    assert isinstance(fit_history, dd.DataFrame)
    assert list(fit_history.columns) == ["fitness", "generation", "individual"]
    assert list(fit_history["fitness"].compute()) == pop_fitness
    assert list(fit_history["generation"].unique().compute()) == [0]
    assert list(fit_history["individual"].compute()) == list(range(size))

    os.system("rm -r out")
