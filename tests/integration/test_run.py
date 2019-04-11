""" Test the algorithm as a whole. """

import os

import dask.dataframe as dd
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


@settings(deadline=None, max_examples=15)
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
    """ Verify that the algorithm produces a valid population history, and keeps
    track of their fitnesses correctly when keeping everything in memory. """

    families = [Normal, Poisson, Uniform]
    for family in families:
        family.reset()

    pop_history, fit_history = edo.run_algorithm(
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

    assert isinstance(fit_history, pd.DataFrame)
    assert all(fit_history.columns == ["fitness", "generation", "individual"])
    assert all(fit_history.dtypes == [float, int, int])
    assert list(fit_history["generation"].unique()) == list(range(max_iter + 1))
    assert list(fit_history["individual"].unique()) == list(range(size))
    assert len(fit_history) % size == 0

    for generation in pop_history:
        assert len(generation) == size

        for individual in generation:
            dataframe, metadata = individual

            assert isinstance(individual, Individual)
            assert isinstance(metadata, list)
            assert isinstance(dataframe, pd.DataFrame)
            assert len(metadata) == len(dataframe.columns)

            for pdf in metadata:
                assert pdf["name"] in [family.name for family in families]


@settings(deadline=None, max_examples=15)
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
    """ Verify that the algorithm produces a valid population history, and keeps
    track of their fitnesses correctly, even when using multiple cores. """

    families = [Normal, Poisson, Uniform]
    for family in families:
        family.reset()

    pop_history, fit_history = edo.run_algorithm(
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
        processes=4,
        fitness_kwargs={"arg": None},
    )

    assert isinstance(fit_history, pd.DataFrame)
    assert all(fit_history.columns == ["fitness", "generation", "individual"])
    assert all(fit_history.dtypes == [float, int, int])
    assert list(fit_history["generation"].unique()) == list(range(max_iter + 1))
    assert list(fit_history["individual"].unique()) == list(range(size))
    assert len(fit_history) % size == 0

    for generation in pop_history:
        assert len(generation) == size

        for individual in generation:
            dataframe, metadata = individual

            assert isinstance(individual, Individual)
            assert isinstance(metadata, list)
            assert isinstance(dataframe, pd.DataFrame)
            assert len(metadata) == len(dataframe.columns)

            for pdf in metadata:
                assert pdf["name"] in [family.name for family in families]


@settings(deadline=None, max_examples=15)
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
def test_run_algorithm_on_disk(
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
    """ Verify that the algorithm produces a valid population history on disk,
    and keeps track of their fitness correctly. """

    families = [Normal, Poisson, Uniform]
    for family in families:
        family.reset()

    pop_history, fit_history = edo.run_algorithm(
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
        root="out",
        processes=4,
        fitness_kwargs={"arg": None},
    )

    assert isinstance(fit_history, dd.DataFrame)
    assert list(fit_history.columns) == ["fitness", "generation", "individual"]
    assert list(fit_history.dtypes) == [float, int, int]
    assert list(fit_history["generation"].unique().compute()) == list(
        range(max_iter + 1)
    )
    assert list(fit_history["individual"].unique().compute()) == list(
        range(size)
    )

    for generation in pop_history:
        assert len(generation) == size

        for individual in generation:
            dataframe, metadata = individual

            assert isinstance(individual, Individual)
            assert isinstance(metadata, list)
            assert isinstance(dataframe, dd.DataFrame)
            assert len(metadata) == len(dataframe.columns)

            for pdf in metadata:
                assert pdf["name"] in [family.name for family in families]

    os.system("rm -r out")
