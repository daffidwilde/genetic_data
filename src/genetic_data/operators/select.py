""" Function(s) for the selection operator. """

from copy import deepcopy

import numpy as np


def selection(population, pop_fitness, best_prop, lucky_prop, maximise):
    """ Given a population, select a proportion of the `best` individuals and
    another of the `lucky` individuals (if they are available) to form a set of
    potential parents. This mirrors the survival of the fittest paradigm whilst
    including a number of less-fit individuals to stop the algorithm from
    converging too early on a suboptimal population. """

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
