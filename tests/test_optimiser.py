""" Tests for the `DataOptimiser` class. """

import itertools as it
import os

import dask.dataframe as dd
import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis.strategies import (
    booleans,
    floats,
    integers,
    lists,
    sampled_from,
    tuples,
)

import edo
from edo import DataOptimiser
from edo.families import all_families
from edo.individual import Individual

from .util.trivials import trivial_fitness

LIMITS = (
    tuples(integers(1, 3), integers(1, 3))
    .map(sorted)
    .filter(lambda x: x[0] <= x[1])
)

OPTIMISER = given(
    size=integers(min_value=2, max_value=5),
    row_limits=LIMITS,
    col_limits=LIMITS,
    families=lists(
        sampled_from(all_families), min_size=2, max_size=2, unique=True
    ),
    weights=sampled_from(
        [
            dist
            for dist in it.product(np.linspace(0.01, 1, 100), repeat=2)
            if sum(dist) == 1.0
        ]
    ),
    max_iter=integers(1, 3),
    best_prop=floats(0.5, 1),
    lucky_prop=floats(0, 1),
    crossover_prob=floats(0, 1),
    mutation_prob=floats(0, 0.5),
    shrinkage=floats(0, 1),
    maximise=booleans(),
)


@OPTIMISER
def test_init(
    size,
    row_limits,
    col_limits,
    families,
    weights,
    max_iter,
    best_prop,
    lucky_prop,
    crossover_prob,
    mutation_prob,
    shrinkage,
    maximise,
):
    """ Test that the `DataOptimiser` class can be instantiated correctly. """

    do = DataOptimiser(
        trivial_fitness,
        size,
        row_limits,
        col_limits,
        families,
        weights,
        max_iter,
        best_prop,
        lucky_prop,
        crossover_prob,
        mutation_prob,
        shrinkage,
        maximise,
    )

    assert do.fitness is trivial_fitness
    assert do.size == size
    assert do.row_limits == row_limits
    assert do.col_limits == col_limits
    assert do.families == families
    assert do.weights == weights
    assert do.max_iter == max_iter
    assert do.best_prop == best_prop
    assert do.lucky_prop == lucky_prop
    assert do.crossover_prob == crossover_prob
    assert do.mutation_prob == mutation_prob
    assert do.shrinkage == shrinkage
    assert do.maximise is maximise

    assert do.converged is False
    assert do.generation == 0
    assert do.population is None
    assert do.pop_fitness is None
    assert do.pop_history is None
    assert do.fit_history is None

    assert edo.cache == {}


@OPTIMISER
def test_stop(
    size,
    row_limits,
    col_limits,
    families,
    weights,
    max_iter,
    best_prop,
    lucky_prop,
    crossover_prob,
    mutation_prob,
    shrinkage,
    maximise,
):
    """ Test that the default stopping method does nothing. """

    do = DataOptimiser(
        trivial_fitness,
        size,
        row_limits,
        col_limits,
        families,
        weights,
        max_iter,
        best_prop,
        lucky_prop,
        crossover_prob,
        mutation_prob,
        shrinkage,
        maximise,
    )

    do.stop()
    assert do.converged is False

    do.converged = "foo"
    do.stop()
    assert do.converged == "foo"


@OPTIMISER
def test_dwindle(
    size,
    row_limits,
    col_limits,
    families,
    weights,
    max_iter,
    best_prop,
    lucky_prop,
    crossover_prob,
    mutation_prob,
    shrinkage,
    maximise,
):
    """ Test that the default dwindling method does nothing. """

    do = DataOptimiser(
        trivial_fitness,
        size,
        row_limits,
        col_limits,
        families,
        weights,
        max_iter,
        best_prop,
        lucky_prop,
        crossover_prob,
        mutation_prob,
        shrinkage,
        maximise,
    )

    do.dwindle()
    assert do.mutation_prob == mutation_prob

    do.mutation_prob = "foo"
    do.dwindle()
    assert do.mutation_prob == "foo"


@OPTIMISER
def test_initialise_run(
    size,
    row_limits,
    col_limits,
    families,
    weights,
    max_iter,
    best_prop,
    lucky_prop,
    crossover_prob,
    mutation_prob,
    shrinkage,
    maximise,
):
    """ Test that the EA can be initialised. """

    do = DataOptimiser(
        trivial_fitness,
        size,
        row_limits,
        col_limits,
        families,
        weights,
        max_iter,
        best_prop,
        lucky_prop,
        crossover_prob,
        mutation_prob,
        shrinkage,
        maximise,
    )

    do._initialise_run(4, None)
    assert isinstance(do.population, list)
    assert len(do.population) == len(do.pop_fitness) == size

    for individual, fitness in zip(do.population, do.pop_fitness):
        assert isinstance(individual, Individual)
        assert isinstance(fitness, float)


@OPTIMISER
@settings(deadline=None, max_examples=10)
def test_run_serial(
    size,
    row_limits,
    col_limits,
    families,
    weights,
    max_iter,
    best_prop,
    lucky_prop,
    crossover_prob,
    mutation_prob,
    shrinkage,
    maximise,
):
    """ Test that the EA can be run serially to produce valid histories. """

    do = DataOptimiser(
        trivial_fitness,
        size,
        row_limits,
        col_limits,
        families,
        weights,
        max_iter,
        best_prop,
        lucky_prop,
        crossover_prob,
        mutation_prob,
        shrinkage,
        maximise,
    )

    pop_history, fit_history = do.run(seed=size, kwargs={"arg": None})

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


@OPTIMISER
@settings(deadline=None, max_examples=10)
def test_run_parallel(
    size,
    row_limits,
    col_limits,
    families,
    weights,
    max_iter,
    best_prop,
    lucky_prop,
    crossover_prob,
    mutation_prob,
    shrinkage,
    maximise,
):
    """ Test that the EA can be run in parallel to produce valid histories. """

    do = DataOptimiser(
        trivial_fitness,
        size,
        row_limits,
        col_limits,
        families,
        weights,
        max_iter,
        best_prop,
        lucky_prop,
        crossover_prob,
        mutation_prob,
        shrinkage,
        maximise,
    )

    pop_history, fit_history = do.run(
        processes=4, seed=size, kwargs={"arg": None}
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


@OPTIMISER
@settings(deadline=None, max_examples=10)
def test_run_on_disk_serial(
    size,
    row_limits,
    col_limits,
    families,
    weights,
    max_iter,
    best_prop,
    lucky_prop,
    crossover_prob,
    mutation_prob,
    shrinkage,
    maximise,
):
    """ Test that the EA can be run with histories on disk and in parallel. """

    do = DataOptimiser(
        trivial_fitness,
        size,
        row_limits,
        col_limits,
        families,
        weights,
        max_iter,
        best_prop,
        lucky_prop,
        crossover_prob,
        mutation_prob,
        shrinkage,
        maximise,
    )

    pop_history, fit_history = do.run(
        root="out", seed=size, kwargs={"arg": None}
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


@OPTIMISER
@settings(deadline=None, max_examples=10)
def test_run_on_disk_parallel(
    size,
    row_limits,
    col_limits,
    families,
    weights,
    max_iter,
    best_prop,
    lucky_prop,
    crossover_prob,
    mutation_prob,
    shrinkage,
    maximise,
):
    """ Test that the EA can be run with histories on disk and serially. """

    do = DataOptimiser(
        trivial_fitness,
        size,
        row_limits,
        col_limits,
        families,
        weights,
        max_iter,
        best_prop,
        lucky_prop,
        crossover_prob,
        mutation_prob,
        shrinkage,
        maximise,
    )

    pop_history, fit_history = do.run(
        root="out", processes=4, seed=size, kwargs={"arg": None}
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
