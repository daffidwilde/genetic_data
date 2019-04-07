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


@settings(deadline=None, max_examples=30)
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
def test_run_algorithm_serial(
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

    families = [Normal, Poisson, Uniform]
    for family in families:
        family.reset()

    _, _, pop_history, fit_history = edo.run_algorithm(
        fitness=trivial_fitness,
        size=size,
        row_limits=row_limits,
        col_limits=col_limits,
        families=families,
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

    for population, pop_fitness in zip(pop_history, fit_history):
        assert len(population) == size
        assert len(pop_fitness) == size

        for individual, fit in zip(population, pop_fitness):

            assert isinstance(fit, float)

            dataframe, metadata = individual

            assert isinstance(individual, Individual)
            assert isinstance(metadata, list)
            assert isinstance(dataframe, pd.DataFrame)
            assert len(metadata) == len(dataframe.columns)

            for pdf in metadata:
                assert (
                    sum([pdf.name == family.name for family in families]) == 1
                )

            for i, limits in enumerate([row_limits, col_limits]):
                assert limits[0] <= dataframe.shape[i] <= limits[1]


@settings(deadline=None, max_examples=30)
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
def test_run_algorithm_parallel(
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

    pass
