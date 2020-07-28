""" A collection of trivial objects for use in tests. """

from edo.individual import Individual


def trivial_fitness(individual, arg=None):
    """ A fitness function. """

    assert isinstance(individual, Individual)
    return 0.0


def trivial_stop(pop_fitness):
    """ A stopping condition. """

    assert isinstance(pop_fitness, list)
    return False
