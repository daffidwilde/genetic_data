""" A script containing functions for each of the components of the genetic
algorithm. """

import numpy as np
import pandas as pd

from copy import deepcopy
from genetic_data.operators import crossover, mutation


def create_individual(row_limits, col_limits, pdfs, weights=None):
    """ Create an individual dataset's allele representation within the limits
    provided. Alleles are given in the form of a tuple.

    Parameters
    ----------
    row_limits : list
        Lower and upper bounds on the number of rows a dataset can have.
    col_limits : list
        Lower and upper bounds on the number of columns a dataset can have.
    pdfs : list
        A list of potential column pdf classes to select from such as those
        found in `genetic_data.pdfs`.
    weights : list
        A sequence of relative weights the same length as `column_classes`. This
        acts as a loose probability distribution from which to sample column
        classes. If `None`, column classes are sampled equally.
    """

    nrows = np.random.randint(row_limits[0], row_limits[1] + 1)
    ncols = np.random.randint(col_limits[0], col_limits[1] + 1)

    individual = pd.DataFrame(
        {
            f"col_{i}": pdf().sample(nrows)
            for i, pdf in enumerate(
                np.random.choice(pdfs, p=weights, size=ncols)
            )
        }
    )

    return individual


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
            "There must be more than one individual in a \
                          population"
        )

    population = [
        create_individual(row_limits, col_limits, pdfs, weights)
        for _ in range(size)
    ]
    return population


def get_fitness(fitness, population):
    """ Return the fitness score of each individual in a population. """

    pop_fitness = [fitness(individual) for individual in population]
    return pop_fitness


def select_parents(population, pop_fitness, best_prop, lucky_prop, maximise):
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


def create_offspring(
    parents,
    size,
    crossover_prob,
    mutation_prob,
    row_limits,
    col_limits,
    pdfs,
    weights,
    sigma,
):
    """ Given a set of potential parents, create offspring from pairs until
    there are enough offspring. Each individual offspring is formed using a
    crossover operator on the two parent individuals and then mutating them
    according to the probability `mutation_prob`. """

    population = []
    while len(population) < size:
        parent1_idx, parent2_idx = np.random.choice(len(parents), size=2)
        parent1, parent2 = parents[parent1_idx], parents[parent2_idx]
        offspring = crossover(parent1, parent2, crossover_prob, pdfs, weights)
        mutant = mutation(
            offspring,
            mutation_prob,
            row_limits,
            col_limits,
            pdfs,
            weights,
            sigma,
        )
        population.append(mutant)

    return population
