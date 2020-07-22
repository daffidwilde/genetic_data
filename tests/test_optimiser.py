""" Tests for the `DataOptimiser` class. """

import itertools as it
import os
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
import yaml
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
from edo.optimiser import _get_fit_history, _get_pop_history

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
def test_get_next_generation(
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
    """ Test that the EA can find the next generation. """

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
    do._get_next_generation(4, {})
    assert isinstance(do.population, list)
    assert len(do.population) == len(do.pop_fitness) == size

    for individual, fitness in zip(do.population, do.pop_fitness):
        assert isinstance(individual, Individual)
        assert isinstance(fitness, float)


@OPTIMISER
def test_update_pop_history(
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
    """ Test that the DataOptimiser can update its population history. """

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
    assert do.pop_history is None

    do._update_pop_history()
    assert len(do.pop_history) == 1
    assert len(do.pop_history[0]) == size
    for i, individual in enumerate(do.population):
        hist_ind = do.pop_history[0][i]
        assert hist_ind.dataframe.equals(individual.dataframe)
        assert hist_ind.metadata == individual.to_history().metadata


@OPTIMISER
def test_update_fit_history(
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
    """ Test that the DataOptimiser can update its fitness history. """

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
    assert do.fit_history is None

    do._update_fit_history()
    fit_history = do.fit_history
    assert fit_history.shape == (size, 3)
    assert list(fit_history.columns) == ["fitness", "generation", "individual"]
    assert list(fit_history["fitness"].values) == do.pop_fitness
    assert list(fit_history["generation"].unique()) == [0]
    assert list(fit_history["individual"]) == list(range(size))

    do.generation += 1
    do._update_fit_history()
    fit_history = do.fit_history
    assert fit_history.shape == (size * 2, 3)
    assert list(fit_history["fitness"].values) == do.pop_fitness * 2
    assert list(fit_history["generation"].unique()) == [0, 1]
    assert list(fit_history["individual"]) == list(range(size)) * 2


@OPTIMISER
def test_update_subtypes(
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
    """ Test that the DataOptimiser can update the subtypes present. """

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
    parents = do.population[: max(int(size / 5), 1)]
    parent_subtypes = {
        type(pdf) for parent in parents for pdf in parent.metadata
    }

    do._update_subtypes(parents)
    updated_subtypes = {
        sub for family in do.families for sub in family.subtypes
    }

    assert parent_subtypes == updated_subtypes


@OPTIMISER
@settings(max_examples=30)
def test_write_generation_serial(
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
    """ Test that the DataOptimiser can write a generation and its fitness to
    file with a single core. """

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
    do._write_generation(root="out", processes=None)
    path = Path("out")

    assert (path / "fitness.csv").exists()
    fit = pd.read_csv(path / "fitness.csv")
    assert list(fit.columns) == ["fitness", "generation", "individual"]
    assert list(fit.dtypes) == [float, int, int]
    assert list(fit["generation"].unique()) == [0]
    assert list(fit["individual"]) == list(range(size))
    assert np.allclose(fit["fitness"].values, do.pop_fitness)

    path /= "0"
    for i, ind in enumerate(do.population):
        ind_path = path / str(i)
        assert (ind_path / "main.csv").exists()
        assert (ind_path / "main.meta").exists()

        df = pd.read_csv(ind_path / "main.csv")
        with open(ind_path / "main.meta", "r") as meta_file:
            meta = yaml.load(meta_file, Loader=yaml.FullLoader)

        assert np.allclose(df.values, ind.dataframe.values)
        assert meta == [m.to_dict() for m in ind.metadata]

    os.system("rm -r out")


@OPTIMISER
@settings(max_examples=30)
def test_write_generation_parallel(
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
    """ Test that the DataOptimiser can write a generation and its fitness to
    file using multiple cores in parallel. """

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
    do._write_generation(root="out", processes=None)
    path = Path("out")

    assert (path / "fitness.csv").exists()
    fit = pd.read_csv(path / "fitness.csv")
    assert list(fit.columns) == ["fitness", "generation", "individual"]
    assert list(fit.dtypes) == [float, int, int]
    assert list(fit["generation"].unique()) == [0]
    assert list(fit["individual"]) == list(range(size))
    assert np.allclose(fit["fitness"].values, do.pop_fitness)

    path /= "0"
    for i, ind in enumerate(do.population):
        ind_path = path / str(i)
        assert (ind_path / "main.csv").exists()
        assert (ind_path / "main.meta").exists()

        df = pd.read_csv(ind_path / "main.csv")
        with open(ind_path / "main.meta", "r") as meta_file:
            meta = yaml.load(meta_file, Loader=yaml.FullLoader)

        assert np.allclose(df.values, ind.dataframe.values)
        assert meta == [m.to_dict() for m in ind.metadata]

    os.system("rm -r out")


@OPTIMISER
@settings(max_examples=10)
def test_get_pop_history(
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
    """ Test that the DataOptimiser can get the population history on disk. """

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
    do._write_generation(root="out", processes=4)

    pop_history = _get_pop_history("out", 1)
    assert isinstance(pop_history, list)
    for generation in pop_history:

        assert isinstance(generation, list)
        for i, individual in enumerate(generation):

            pop_ind = do.population[i]
            assert isinstance(individual.dataframe, dd.DataFrame)
            assert np.allclose(
                pop_ind.dataframe.values, individual.dataframe.values.compute()
            )
            assert isinstance(individual.metadata, list)
            assert individual.metadata == [
                m.to_dict() for m in pop_ind.metadata
            ]

    os.system("rm -r out")


@OPTIMISER
@settings(max_examples=10)
def test_get_fit_history(
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
    """ Test that the DataOptimiser can get the fitness hsitory on disk. """

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
    do._write_generation(root="out", processes=4)

    fit_history = _get_fit_history("out")
    assert isinstance(fit_history, dd.DataFrame)
    assert list(fit_history.columns) == ["fitness", "generation", "individual"]
    assert list(fit_history["fitness"].compute()) == do.pop_fitness
    assert list(fit_history["generation"].unique().compute()) == [0]
    assert list(fit_history["individual"].compute()) == list(range(size))

    os.system("rm -r out")


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


@OPTIMISER
@settings(deadline=None, max_examples=10)
def test_run_is_reproducible(
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
    """ Test that two runs of the EA with the same parameters produce the
    same population and fitness histories. """

    do_one = DataOptimiser(
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

    pop_history_one, fit_history_one = do_one.run(
        processes=4, seed=size, kwargs={"arg": None}
    )

    do_two = DataOptimiser(
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

    pop_history_two, fit_history_two = do_two.run(
        processes=4, seed=size, kwargs={"arg": None}
    )

    assert fit_history_one.equals(fit_history_two)

    for gen_from_one, gen_from_two in zip(pop_history_one, pop_history_two):
        for ind_from_one, ind_from_two in zip(gen_from_one, gen_from_two):
            assert ind_from_one.dataframe.equals(ind_from_two.dataframe)
