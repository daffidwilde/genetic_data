""" A script containing functions for each of the components of the genetic
algorithm. """

import numpy as np

from .individual import create_individual
from .operators import crossover, mutation


def create_initial_population(size, row_limits, col_limits, pdfs, weights=None):
    """ Create an initial population for the genetic algorithm based on the
    given parameters.

    Parameters
    ----------
    size : int
        The number of individuals in the population.
    row_limits : list
        Limits on the number of rows a dataset can have.
    col_limits : list
        Limits on the number of columns a dataset can have.
    pdfs : list
        A list of potential column pdf classes such as those found in
        `pdfs.py`. Must have a `.sample()` and `.mutate()` method.
    weights : list
        A sequence of relative weights the same length as `column_classes`. This
        acts as a loose probability distribution from which to sample column
        classes. If `None`, column classes are sampled equally.

    Returns
    -------
    population : list
        A collection of individuals.
    """

    if size <= 1:
        raise ValueError(
            "There must be more than one individual in a population"
        )

    population = [
        create_individual(row_limits, col_limits, pdfs, weights)
        for _ in range(size)
    ]

    return population


def create_new_population(
    parents,
    size,
    crossover_prob,
    mutation_prob,
    row_limits,
    col_limits,
    pdfs,
    weights,
):
    """ Given a set of potential parents to be carried into the next generation,
    create offspring from pairs within that set until there are enough
    individuals. Each individual offspring is formed using a crossover operator
    on the two parent individuals and then mutating them according to the
    probability `mutation_prob`. """

    population = parents
    while len(population) < size:
        parent1_idx, parent2_idx = np.random.choice(len(parents), size=2)
        parents_ = parents[parent1_idx], parents[parent2_idx]
        offspring = crossover(*parents_, col_limits, pdfs, crossover_prob)
        mutant = mutation(
            offspring, mutation_prob, row_limits, col_limits, pdfs, weights
        )
        for col, meta in zip(*mutant):
            mutant.dataframe[col] = mutant.dataframe[col].astype(meta.dtype)
        population.append(mutant)

    return population
