""" Test the algorithm as a whole. """

import pandas as pd
from hypothesis import given, settings
from hypothesis.strategies import booleans

import edo
from edo.individual import Individual
from edo.pdfs import Normal, Poisson, Uniform

from .util.parameters import PROB, SHAPES, SIZE, WEIGHTS
from .util.trivials import trivial_dwindle, trivial_fitness, trivial_stop

HALF_PROB = PROB.filter(lambda x: x > 0.5)
OPEN_UNIT = PROB.filter(lambda x: x not in [0, 1])


@settings(deadline=None)
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
    shrinkage=OPEN_UNIT,
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
    shrinkage,
    maximise,
    seed,
):
    """ Verify that the algorithm produces a valid population, and keeps track
    of them/their fitnesses correctly. """

    pdfs = [Normal, Poisson, Uniform]

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
        shrinkage=shrinkage,
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
                assert limits[0] <= dataframe.shape[i] <= limits[1]

            for score in scores:
                assert isinstance(score, float)
