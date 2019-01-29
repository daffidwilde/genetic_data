""" A collection of trivial objects for use in tests. """

import numpy as np
import pandas as pd


def trivial_fitness(individual, arg=None):
    """ A fitness function. """
    assert isinstance(individual, pd.DataFrame)
    assert arg is None
    return np.nan


def trivial_stop(pop_fitness):
    """ A stopping condition. """
    assert isinstance(pop_fitness, list)
    return False
