""" Tests for the writing of individuals to file. """

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from hypothesis import given, settings
from hypothesis.strategies import integers

from edo.individual import create_individual
from edo.pdfs import Normal, Poisson, Uniform
from edo.population import create_initial_population
from edo.write import write_fitness, write_generation, write_individual

from .util.parameters import INTEGER_INDIVIDUAL, POPULATION
from .util.trivials import trivial_fitness


@INTEGER_INDIVIDUAL
def test_write_individual(row_limits, col_limits, weights):
    """ Test that an individual can be saved in the correct place. """

    families = [Normal, Poisson, Uniform]
    individual = create_individual(row_limits, col_limits, families, weights)

    write_individual(individual, gen=0, idx=0, root="out").compute()
    path = Path("out/0/0")
    assert (path / "main.csv").exists()
    assert (path / "main.meta").exists()

    df = pd.read_csv(path / "main.csv")
    with open(path / "main.meta", "r") as meta_file:
        meta = yaml.load(meta_file)

    assert np.allclose(df.values, individual.dataframe.values)
    assert meta == [m.to_dict() for m in individual.metadata]

    os.system("rm -r out")


@given(size=integers(min_value=1, max_value=100))
def test_write_fitness(size):
    """ Test that a generation's fitness can be written to file correctly. """

    fitness = [trivial_fitness(pd.DataFrame()) for _ in range(size)]

    write_fitness(fitness, gen=0, root="out").compute()
    write_fitness(fitness, gen=1, root="out").compute()
    path = Path("out")
    assert (path / "fitness.csv").exists()

    fit = pd.read_csv(path / "fitness.csv")
    assert list(fit.columns) == ["fitness", "generation", "individual"]
    assert list(fit.dtypes) == [float, int, int]
    assert list(fit["generation"].unique()) == [0, 1]
    assert list(fit["individual"]) == list(range(size)) * 2
    assert np.allclose(fit["fitness"].values, fitness * 2)


@POPULATION
@settings(max_examples=30)
def test_write_generation_serial(size, row_limits, col_limits, weights):
    """ Test that an entire generation and its fitness can be written to file
    correctly using a single core. """

    families = [Normal, Poisson, Uniform]
    population = create_initial_population(
        size, row_limits, col_limits, families, weights
    )
    fitness = [trivial_fitness(ind.dataframe) for ind in population]

    write_generation(population, fitness, gen=0, root="out")
    path = Path("out")

    assert (path / "fitness.csv").exists()
    fit = pd.read_csv(path / "fitness.csv")
    assert list(fit.columns) == ["fitness", "generation", "individual"]
    assert list(fit.dtypes) == [float, int, int]
    assert list(fit["generation"].unique()) == [0]
    assert list(fit["individual"]) == list(range(size))
    assert np.allclose(fit["fitness"].values, fitness)

    path /= "0"
    for i, ind in enumerate(population):
        ind_path = path / str(i)
        assert (ind_path / "main.csv").exists()
        assert (ind_path / "main.meta").exists()

        df = pd.read_csv(ind_path / "main.csv")
        with open(ind_path / "main.meta", "r") as meta_file:
            meta = yaml.load(meta_file)

        assert np.allclose(df.values, ind.dataframe.values)
        assert meta == [m.to_dict() for m in ind.metadata]

    os.system("rm -r out")


@POPULATION
@settings(max_examples=30)
def test_write_generation_parallel(size, row_limits, col_limits, weights):
    """ Test that an entire generation and its fitness can be written to file
    correctly using multiple cores in parallel. """

    families = [Normal, Poisson, Uniform]
    population = create_initial_population(
        size, row_limits, col_limits, families, weights
    )
    fitness = [trivial_fitness(ind.dataframe) for ind in population]

    write_generation(population, fitness, gen=0, root="out", processes=4)
    path = Path("out")

    assert (path / "fitness.csv").exists()
    fit = pd.read_csv(path / "fitness.csv")
    assert list(fit.columns) == ["fitness", "generation", "individual"]
    assert list(fit.dtypes) == [float, int, int]
    assert list(fit["generation"].unique()) == [0]
    assert list(fit["individual"]) == list(range(size))
    assert np.allclose(fit["fitness"].values, fitness)

    path /= "0"
    for i, ind in enumerate(population):
        ind_path = path / str(i)
        assert (ind_path / "main.csv").exists()
        assert (ind_path / "main.meta").exists()

        df = pd.read_csv(ind_path / "main.csv")
        with open(ind_path / "main.meta", "r") as meta_file:
            meta = yaml.load(meta_file)

        assert np.allclose(df.values, ind.dataframe.values)
        assert meta == [m.to_dict() for m in ind.metadata]

    os.system("rm -r out")
