""" Tests for the writing of individuals to file. """

import os

from pathlib import Path

import numpy as np
import pandas as pd

from edo.run import write_individual
from edo.individual import create_individual
from edo.pdfs import Normal, Poisson, Uniform

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

    os.system("rm -r out")
