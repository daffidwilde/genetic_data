""" The main script containing a generic genetic algorithm. """

import random

import numpy as np

from genetic_data.components import create_initial_population, \
                                    create_offspring, get_fitness, \
                                    get_ordered_population, select_parents, \
                                    mutate_population

def run_algorithm(fitness, size, row_limits, col_limits, pdfs, weights=None,
                  num_samples=10, amalgamation_method=np.mean, stop=None,
                  max_iter=100, best_prop=0.25, lucky_prop=0.01, prob=0.5,
                  mutation_prob=1, allele_prob=0.01, seed=0):
    """ Run a genetic algorithm (GA) under the presented constraints, giving a
    population of artificial datasets for which the fitness function performs
    well.

    Parameters
    ----------
    fitness : func
        Any real-valued function that takes one `pandas.DataFrame` as argument.
        Here, higher values are considered 'fitter'.
    size : int
        The size of the population to create.
    row_limits : list
        Lower and upper bounds on the number of rows a dataset can have.
    col_limits : list
        Lower and upper bounds on the number of columns a dataset can have.
    pdfs : list
        The classes of distribution each column of a dataset can take. These
        distributions should take values either from the real numbers or the
        integers. Also, classes must have `sample` and `mutate` methods
        detailing how to sample from the distribution, and how to mutate itself
        and its parameters.
    weights : list
        Relative probability distribution on how to select columns from
        `column_classes`. If not specified, will choose evenly.
    num_samples : int
        The number of samples to take from an individual's dataset family. These
        samples are used to determine the fitness of the individual.
    amalgamation_method : func
        How to amalgamate the fitness of the samples taken from an individual's
        dataset family. By default, the mean is taken using `np.mean` but any
        function may be used that returns a real value.
    stop : float
        A stopping tolerance on the coefficient of variation in the fitness of
        the current generation. If `None`, the GA will run to its maximum number
        of iterations.
    max_iter : int
        The maximum number of iterations to be carried out before stopping.
    best_prop : float
        The proportion of a population from which to select the "best" potential
        parents.
    lucky_prop : float
        The proportion of a population from which to sample some "lucky"
        potential parents.
    prob : float
        The probability with which to sample from the first parent over the
        second in a crossover operation.
    mutation_prob : float
        The probability of any particular individual being mutated in a
        generation.
    allele_prob : float
        The probability of any particular allele in an individual being mutated.
    seed : int
        The seed for a pseudo-random number generator for the run of the
        algorithm.
    """

    random.seed(seed)

    for pdf in pdfs:
        pdf.alt_pdfs = [p for p in pdfs if p != pdf]

    population = create_initial_population(size, row_limits, col_limits,
                                           pdfs, weights)
    pop_fitness = get_fitness(fitness, population, num_samples,
                              amalgamation_method)

    if stop:
        converged = abs(np.std(pop_fitness) / np.mean(pop_fitness)) < stop
    else:
        converged = False

    all_fitness_scores = [pop_fitness]
    itr = 0
    while itr < max_iter and not converged:
        parents = select_parents(population, pop_fitness, best_prop, lucky_prop)
        offspring = create_offspring(parents, prob, size)

        population = mutate_population(offspring, mutation_prob, allele_prob,
                                       row_limits, col_limits, pdfs, weights)
        pop_fitness = get_fitness(fitness, population, num_samples,
                                  amalgamation_method)
        all_fitness_scores.append(pop_fitness)
        if stop:
            converged = abs(np.std(pop_fitness) / np.mean(pop_fitness)) < stop
        itr += 1

    return population, pop_fitness, all_fitness_scores
