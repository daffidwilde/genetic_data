""" Tests for each of the steps in the population generation process. """

import pandas as pd
import numpy as np
import pytest

from hypothesis import settings

from genetic_data.components import (
    create_initial_population,
    select_parents,
    get_fitness,
    create_offspring,
)
from genetic_data.pdfs import Gamma, Normal, Poisson

from test_util.parameters import OFFSPRING, SELECTION, SMALL_PROPS
from test_util.trivials import trivial_fitness


@SELECTION
def test_select_parents(size, row_limits, col_limits, weights, props, maximise):
    """ Create a population, get its fitness and select potential parents
    based on that fitness. Verify that parents are all valid individuals. """

    best_prop, lucky_prop = props
    pdfs = [Gamma, Normal, Poisson]
    population = create_initial_population(
        size, row_limits, col_limits, pdfs, weights
    )
    pop_fitness = get_fitness(trivial_fitness, population)
    parents = select_parents(
        population, pop_fitness, best_prop, lucky_prop, maximise
    )

    assert len(parents) == min(
        size, int(best_prop * size) + int(lucky_prop * size)
    )

    for ind in parents:
        assert isinstance(ind, pd.DataFrame)


@SMALL_PROPS
def test_select_parents_raises_error(
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

        select_parents(population, pop_fitness, best_prop, lucky_prop, maximise)


@OFFSPRING
@settings(max_examples=25, deadline=None)
def test_create_offspring(
    size,
    row_limits,
    col_limits,
    weights,
    props,
    crossover_prob,
    mutation_prob,
    maximise,
    sigma,
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
    parents = select_parents(
        population, pop_fitness, best_prop, lucky_prop, maximise
    )

    population = create_offspring(
        parents,
        size,
        crossover_prob,
        mutation_prob,
        row_limits,
        col_limits,
        pdfs,
        weights,
        sigma,
    )
    assert isinstance(population, list)
    assert len(population) == size

    for parent in parents:
        try:
            assert np.any([np.all(parent == ind) for ind in population])
        except:
            ValueError

    for ind in population:
        assert isinstance(ind, pd.DataFrame)
