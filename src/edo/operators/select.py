""" .. Function(s) for the selection operator. """

from copy import deepcopy

import numpy as np


def selection(population, pop_fitness, best_prop, lucky_prop, maximise=False):
    """ Given a population, select a proportion of the "best"" individuals and
    another of the "lucky" individuals (if they are available) to form a set of
    potential parents.

    Parameters
    ----------
    population : list
        All current individuals.
    pop_fitness : list
        The fitness of each individual in :code:`population`.
    best_prop : float
        The proportion of the fittest individuals in :code:`population` to be
        selected.
    lucky_prop : float
        The proportion of lucky individuals in :code:`population` to be
        selected.
    maximise : bool, optional
        Determines whether an individual's fitness should be minimal or not.
        Defaults to minimise.

    Returns
    -------
    parents : list
        The individuals chosen to potentially become parents.

    Raises
    ------
    ValueError
        If :code:`int(best_prop * len(population)) == 0` and
        :code:`int(lucky_prop * len(population)) == 0`.
    """

    size = len(population)
    num_best = int(best_prop * size)
    num_lucky = int(lucky_prop * size)

    if maximise:
        best_choice = np.argmax
    else:
        best_choice = np.argmin

    if num_best == 0 and num_lucky == 0:
        raise ValueError(
            'Not a large enough proportion of "best" and/or \
                          "lucky" individuals chosen. Reconsider these values.'
        )

    population = deepcopy(population)
    pop_fitness = deepcopy(pop_fitness)
    parents = []
    for _ in range(num_best):
        if population != []:
            best = best_choice(pop_fitness)
            pop_fitness.pop(best)
            parents.append(population.pop(best))

    for _ in range(num_lucky):
        if population != []:
            lucky = np.random.choice(len(population))
            parents.append(population.pop(lucky))

    return parents
