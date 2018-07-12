""" The main script containing a generic genetic algorithm. """

import random

from genetic_data.pdfs import Gamma, Poisson
from genetic_data.components import create_initial_population, \
                                    create_offspring, get_fitness, \
                                    select_parents, mutate_population 

def run_algorithm(fitness, size, row_limits, col_limits,
                  column_classes=[Gamma, Poisson], weights=None, stop=0.01,
                  max_iter=100, best_prop=0.2, lucky_prop=0.05,
                  mutation_rate=0.01, allele_prob=0.1, seed=0):
    """ Run a genetic algorithm under the presented constraints, giving a
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
    column_classes : list
        The classes of distribution each column of a dataset can take. These
        distributions should take values either from the real numbers or the
        integers. Also, classes must have `__str__`, `sample` and `mutate`
        methods detailing the name of the distribution, how to sample from the
        distribution in question, and how to mutate itself and its parameters.
    weights : list
        Relative probability distribution on how to select columns from
        `column_classes`. If not specified, will choose evenly.
    stop : float
        A stopping tolerance on the normed change in the average fitness from
        the last generation.
    max_iter : int
        The maximum number of iterations to be carried out before stopping.
    best_prop : float
        The proportion of a population from which to select the "best" potential
        parents.
    lucky_prop : float
        The proportion of a population from which to sample some "lucky"
        potential parents.
    mutation_rate : float
        The probability of any particular individual being mutated in a
        generation.
    allele_prob : float
        Given an individual that is to be mutated, this is the probability of
        any particular allele in that individual being mutated.
    seed : int
        The seed for a pseudo-random number generator for the run of the
        algorithm.
    """

    random.seed(seed)

    population = create_initial_population(size, row_limits, col_limits,
            column_classes, weights,)
