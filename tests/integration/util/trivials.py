""" A collection of trivial objects for use in tests. """

import numpy as np


def trivial_fitness(individual, arg=None):
    """ A fitness function. """
    return np.nan


def trivial_stop(pop_fitness):
    """ A stopping condition. """
    return False


def trivial_dwindle(mutation_prob, itr):
    """ A dwindling mutation probability function. """
    assert isinstance(mutation_prob, float)
    assert isinstance(itr, int)
    return mutation_prob
