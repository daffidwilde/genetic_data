""" The selection operator. """

import numpy as np


def selection(
    population, pop_fitness, best_prop, lucky_prop, random_state, maximise=False
):
    """ Given a population, select a proportion of the "best" individuals and
    another of the "lucky" individuals (if they are available) to form a set of
    potential parents.

    Parameters
    ----------
    population : list
        All current individuals.
    pop_fitness : list
        The fitness of each individual in ``population``.
    best_prop : float
        The proportion of the fittest individuals in ``population`` to be
        selected.
    lucky_prop : float
        The proportion of lucky individuals left in ``population`` to be
        selected after the "best" have been selected.
    maximise : bool, optional
        Determines whether an individual's fitness should be maximal or not.
        Defaults to ``False``.

    Returns
    -------
    parents : dict
        The individuals chosen to potentially become parents and their index in
        the current population.
    """

    size = len(population)
    num_best = int(best_prop * size)
    num_lucky = int(lucky_prop * size)

    if maximise:
        best_choice = np.argmax
    else:
        best_choice = np.argmin

    population = population[:]
    pop_fitness = pop_fitness[:]
    parents = []
    for _ in range(num_best):
        if population != []:
            best = best_choice(pop_fitness)
            pop_fitness.pop(best)
            parents.append(population.pop(best))

    for _ in range(num_lucky):
        if population != []:
            lucky = random_state.choice(len(population))
            parents.append(population.pop(lucky))

    return parents
