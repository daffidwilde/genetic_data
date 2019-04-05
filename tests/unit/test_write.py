""" Tests for the writing of individuals to file. """

import os
from pathlib import Path

from hypothesis import given, settings
from hypothesis.strategies import integers

import numpy as np
import pandas as pd

from edo.write import write_fitness, write_generation, write_individual
from edo.individual import create_individual
from edo.pdfs import Normal, Poisson, Uniform
from edo.population import create_initial_population

from .util.parameters import INTEGER_INDIVIDUAL, POPULATION
from .util.trivials import trivial_fitness


@INTEGER_INDIVIDUAL
def test_write_individual(row_limits, col_limits, weights):
    """ Test that an individual can be saved in the correct place. """

    families = [Normal, Poisson, Uniform]
    individual = create_individual(row_limits, col_limits, families, weights)

    write_individual(individual, gen=0, idx=0, root="out")
    path = Path("out/0/0")
    assert (path / "main.csv").exists()
    assert (path / "meta.csv").exists()

    df = pd.read_csv(path / "main.csv")
    assert np.allclose(df.values, individual.dataframe.values)


@given(size=integers(min_value=1, max_value=100))
def test_write_fitness(size):
    """ Test that a generation's fitness can be written to file correctly. """

    fitness = [trivial_fitness(pd.DataFrame()) for _ in range(size)]

    write_fitness(fitness, gen=0, root="out")
    path = Path(f"out/0")
    assert (path / "fitness.csv").exists()

    fit = pd.read_csv(path / "fitness.csv")
    assert np.allclose(fit.values.reshape(size,), fitness)


@POPULATION
@settings(max_examples=30)
def test_write_generation(size, row_limits, col_limits, weights):
    """ Test that an entire generation and its fitness can be written to file
    correctly. """

    families = [Normal, Poisson, Uniform]
    population = create_initial_population(size, row_limits, col_limits,
            families, weights)
    fitness = [trivial_fitness(ind.dataframe) for ind in population]

    write_generation(population, fitness, gen=0, root="out")
    path = Path("out/0")
    assert (path / "fitness.csv").exists()
    for i, ind in enumerate(population):
        ind_path = path / str(i)
        assert (ind_path / "main.csv").exists()
        assert (ind_path / "meta.csv").exists()

        df = pd.read_csv(ind_path / "main.csv")
        assert np.allclose(df.values, ind.dataframe.values)

    os.system("rm -r out")
