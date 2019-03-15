""" Tests for the selection operator. """

import pandas as pd
import pytest

from edo.fitness import get_fitness
from edo.individual import Individual
from edo.operators import selection
from edo.pdfs import Gamma, Normal, Poisson
from edo.population import create_initial_population

from .util.parameters import SELECTION, SMALL_PROPS
from .util.trivials import trivial_fitness


@SELECTION
def test_parents(size, row_limits, col_limits, weights, props, maximise):
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
        dataframe, metadata = individual

        assert isinstance(individual, Individual)
        assert isinstance(metadata, list)
        assert isinstance(dataframe, pd.DataFrame)
        assert len(metadata) == len(dataframe.columns)

        for pdf in metadata:
            assert isinstance(pdf, tuple(pdfs))

        for i, limits in enumerate([row_limits, col_limits]):
            assert limits[0] <= dataframe.shape[i] <= limits[1]


@SMALL_PROPS
def test_smallprops_error(
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
