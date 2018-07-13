""" A script containing functions for each of the components of the genetic
algorithm. """

import random
import numpy as np
import pandas as pd

from genetic_data.operators import crossover, mutate_individual

def create_individual(row_limits, col_limits, column_classes, weights=None,
                      alt_pdfs=None):
    """ Create an individual dataset's allele representation within the limits
    provided. Alleles are given in the form of a tuple.

    Parameters
    ----------
    row_limits : list
        Lower and upper bounds on the number of rows a dataset can have.
    col_limits : list
        Lower and upper bounds on the number of columns a dataset can have.
    column_classes : list
        A list of potential column classes to select from such as those found in
        `pdfs.py`.
    weights : list
        A sequence of relative weights the same length as `column_classes`. This
        acts as a loose probability distribution from which to sample column
        classes. If `None`, column classes are sampled equally.
    alt_pdfs : dict
        The name of each class of column pdf acts as a key with its value being
        a list of all the other types of column pdf available.
    """

    nrows = random.randint(*row_limits)
    ncols = random.randint(*col_limits)

    column_pdfs = [col(alt_pdfs) for col in column_classes]

    individual = tuple([
        nrows, ncols, *random.choices(column_pdfs, weights=weights, k=ncols)
    ])

    return individual

def create_initial_population(size, row_limits, col_limits,
                              column_classes, weights=None, alt_pdfs=None):
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
    column_classes : list
        A list of potential column classes such as those found in
        `column_pdfs.py`. Must have a `.sample()` and `.mutate()` method.
    weights : list
        A sequence of relative weights the same length as `column_classes`. This
        acts as a loose probability distribution from which to sample column
        classes. If `None`, column classes are sampled equally.
    alt_pdfs : dict
        The name of each class of column pdf acts as a key with its value being
        a list of all other column pdf types avaiable.

    Returns
    -------
    population : list
        A collection of individuals.
    """

    if size <= 1:
        raise ValueError('There must be more than one individual in a \
                          population')

    population = []
    for _ in range(size):
        individual = create_individual(row_limits, col_limits,
                                       column_classes, weights, alt_pdfs)
        population.append(individual)

    return population

def get_dataframes(individual, max_seed):
    """ Sample `max_seed` datasets from the family of datasets represented by an
    individual's alleles. Each of these is a `pandas.DataFrame` object. """

    dfs = []
    for seed in range(max_seed):
        nrows = individual[0]
        df = pd.DataFrame({f'col_{i}': col.sample(nrows, seed) \
                           for i, col in enumerate(individual[2:])})
        dfs.append(df)

    return dfs

def get_fitness(fitness, population, max_seed, selection=np.mean):
    """ Return the fitness score of each individual in a population according to
    a selection function over a set of fitness scores from seeded samples. By
    default, the mean of these fitness scores is taken. However, any function
    can be passed here on how to choose from the set of fitness scores. Some
    examples could be: choosing the best-case scenario with Python's `min` or
    `max` functions; taking the median score; or, cutting off outliers to give a
    truncated mean. """

    individual_fitnesses = np.empty((len(population), max_seed))
    population_fitness = np.empty(len(population))
    for i, individual in enumerate(population):
        dfs = get_dataframes(individual, max_seed)
        for j, df in enumerate(dfs):
            individual_fitnesses[i, j] = fitness(df)
        population_fitness[i] = selection(individual_fitnesses[i, :])

    return population_fitness

def get_ordered_population(population, population_fitness, maximise=True):
    """ Return a dictionary with key-value pairs given by the individuals in a
    population and their respective fitness. This population is sorted into
    order depending on the contet of the fitness score of an individual. By
    default, the population is ordered into descending order of fitness so that
    higher fitness scores reflect fitter individuals. """

    fitness_dict = {
        ind: fit for ind, fit in zip(population, population_fitness)
    }
    ordered_population = dict(
        sorted(fitness_dict.items(), reverse=maximise, key=lambda x: x[1])
    )

    return ordered_population

def select_parents(ordered_population, best_prop, lucky_prop):
    """ Given a population ranked by their fitness, select a proportion of the
    `best` individuals and another of the `lucky` individuals (if they are
    available) to form a set of potential parents. This mirrors the survival of
    the fittest paradigm whilst including a number of less-fit individuals to
    stop the algorithm from converging too early. """

    size = len(ordered_population)
    num_best = max(int(best_prop * size), 1)
    num_lucky = max(int(lucky_prop * size), 1)
    population = list(ordered_population.keys())

    parents = []
    for _ in range(num_best):
        if population != []:
            best = population.pop(0)
            parents.append(best)

    for _ in range(num_lucky):
        if population != []:
            lucky = random.choice(population)
            parents.append(lucky)
            population.remove(lucky)

    return parents

def create_offspring(parents, prob, size):
    """ Given a set of potential parents, create offspring from pairs until
    there are enough offspring. Each individual offspring is formed using a
    crossover operator on the two parent individuals. """

    offspring = []
    while len(offspring) < size:
        parent1, parent2 = random.choices(parents, k=2)
        child = crossover(parent1, parent2, prob)
        offspring.append(child)

    return offspring

def mutate_population(population, mutation_rate, allele_prob, row_limits,
                      col_limits, pdfs, weights, alt_pdfs):
    """ Given a population, mutate a small number of its individuals according
    to a mutation rate. For each individual that is to be mutated, their alleles
    are mutated with a separate probability `allele_prob`. """

    new_population = []
    for ind in population:
        if random.random() < mutation_rate:
            ind = mutate_individual(ind, allele_prob, row_limits, col_limits,
                                    pdfs, weights, alt_pdfs)
        new_population.append(ind)

    return new_population
