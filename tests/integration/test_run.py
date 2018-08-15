""" Test the algorithm as a whole. """

from hypothesis import given, settings
from hypothesis.strategies import booleans

import pandas as pd

import genetic_data as gd

from genetic_data.individual import Individual
from genetic_data.pdfs import Gamma, Normal, Poisson

from .util.trivials import trivial_fitness, trivial_stop
from .util.parameters import PROB, SHAPES, SIZE, WEIGHTS

HALF_PROB = PROB.filter(lambda x: x > 0.5)


@given(
    size=SIZE,
    row_limits=SHAPES,
    col_limits=SHAPES,
    weights=WEIGHTS,
    max_iter=SIZE,
    best_prop=HALF_PROB,
    lucky_prop=HALF_PROB,
    crossover_prob=PROB,
    mutation_prob=PROB,
    maximise=booleans(),
    seed=SIZE,
)
@settings(deadline=None)
def test_run_algorithm(
    size,
    row_limits,
    col_limits,
    weights,
    max_iter,
    best_prop,
    lucky_prop,
    crossover_prob,
    mutation_prob,
    maximise,
    seed,
):
    """ Verify that the algorithm produces a valid population, and keeps track
    of them/their fitnesses correctly. """

    pdfs = [Gamma, Normal, Poisson]
    stop = trivial_stop

    pop, fit, all_pops, all_fits = gd.run_algorithm(
        trivial_fitness,
        size,
        row_limits,
        col_limits,
        pdfs,
        weights,
        stop,
        max_iter,
        best_prop,
        lucky_prop,
        crossover_prob,
        mutation_prob,
        maximise=maximise,
        seed=seed,
    )

    assert len(pop) == size
    assert len(fit) == size

    for population, scores in zip(all_pops, all_fits):
        assert len(population) == size
        assert len(scores) == size

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

            for score in scores:
                assert isinstance(score, float)
