""" Tests for the crossover and mutation operator scripts. """

import pandas as pd

from hypothesis import settings

from genetic_data.components import create_individual
from genetic_data.operators import crossover, mutation
from genetic_data.pdfs import Gamma, Normal, Poisson

from test_util.parameters import CROSSOVER, MUTATION


@CROSSOVER
@settings(deadline=None)
def test_crossover(row_limits, col_limits, weights, prob):
    """ Verify that `crossover` produces a valid individual. """

    pdfs = [Gamma, Normal, Poisson]
    parents = [
        create_individual(row_limits, col_limits, pdfs, weights) for _ in [0, 1]
    ]

    individual = crossover(*parents, prob, pdfs, weights)

    assert isinstance(individual, pd.DataFrame)
    assert individual.shape[0] in [parent.shape[0] for parent in parents]


@MUTATION
def test_mutation(row_limits, col_limits, weights, prob, sigma):
    """ Verify that `mutation` creates a valid individual. """

    pdfs = [Gamma, Normal, Poisson]

    individual = create_individual(row_limits, col_limits, pdfs, weights)
    mutant = mutation(
        individual, prob, row_limits, col_limits, pdfs, weights, sigma
    )

    assert isinstance(mutant, pd.DataFrame)

    for axis in [0, 1]:
        assert (
            mutant.shape[axis] >= individual.shape[axis]
            or mutant.shape[axis] <= individual.shape[axis]
        )
