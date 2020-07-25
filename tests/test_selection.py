""" Tests for the selection operator. """

import numpy as np
import pandas as pd

from edo import Family
from edo.distributions import Gamma, Normal, Poisson
from edo.fitness import get_population_fitness
from edo.individual import Individual
from edo.operators import selection
from edo.population import create_initial_population

from .util.parameters import SELECTION
from .util.trivials import trivial_fitness


@SELECTION
def test_selection_by_parents(
    size, row_limits, col_limits, weights, props, maximise
):
    """ Create a population, get its fitness and select potential parents
    based on that fitness. Verify that parents are all valid individuals. """

    best_prop, lucky_prop = props
    distributions = [Gamma, Normal, Poisson]
    families = [Family(dist) for dist in distributions]
    states = {i: np.random.RandomState(i) for i in range(size)}
    state = np.random.RandomState(size)

    population = create_initial_population(
        row_limits, col_limits, families, weights, states
    )

    pop_fitness = get_population_fitness(population, trivial_fitness)
    parents = selection(
        population, pop_fitness, best_prop, lucky_prop, state, maximise
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
            assert sum(pdf.family is family for family in families) == 1

        for i, limits in enumerate([row_limits, col_limits]):
            assert limits[0] <= dataframe.shape[i] <= limits[1]
