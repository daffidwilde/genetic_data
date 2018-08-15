""" A collection of trivial objects for use in tests. """

import numpy as np


def trivial_fitness(individual):
    """ A fitness function. """
    return np.nan


def trivial_stop(pop_fitness):
    """ A stopping condition. """
    return False
