""" Test the algorithm as a whole. """

import pandas as pd
from hypothesis import given
from hypothesis.strategies import booleans

import edo
from edo.individual import Individual
from edo.pdfs import Gamma, Normal, Poisson

from .util.parameters import PROB, SHAPES, SIZE, WEIGHTS
from .util.trivials import trivial_dwindle, trivial_fitness, trivial_stop

HALF_PROB = PROB.filter(lambda x: x > 0.5)
OPEN_UNIT = PROB.filter(lambda x: x not in [0, 1])


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
    compact=OPEN_UNIT,
    maximise=booleans(),
    seed=SIZE,
)
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
    compact,
    maximise,
    seed,
):
    """ Verify that the algorithm produces a valid population, and keeps track
    of them/their fitnesses correctly. """

    pdfs = [Gamma, Normal, Poisson]

    for pdf in pdfs: # make sure the classes are as normal
        pdf.reset()

    pop, fit, all_pops, all_fits = edo.run_algorithm(
        fitness=trivial_fitness,
        size=size,
        row_limits=row_limits,
        col_limits=col_limits,
        pdfs=pdfs,
        weights=weights,
        stop=trivial_stop,
        dwindle=trivial_dwindle,
        max_iter=max_iter,
        best_prop=best_prop,
        lucky_prop=lucky_prop,
        crossover_prob=crossover_prob,
        mutation_prob=mutation_prob,
        compact=compact,
        maximise=maximise,
        seed=seed,
        fitness_kwargs={"arg": None},
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
