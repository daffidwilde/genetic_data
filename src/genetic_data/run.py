""" The main script containing a generic genetic algorithm. """

import numpy as np

from genetic_data.pdfs import Normal
from genetic_data.components import (
    create_initial_population,
    create_offspring,
    get_fitness,
    select_parents,
)


def run_algorithm(
    fitness,
    size,
    row_limits,
    col_limits,
    pdfs=[Normal],
    weights=None,
    stop=None,
    max_iter=100,
    best_prop=0.25,
    lucky_prop=0,
    crossover_prob=0.5,
    mutation_prob=0.01,
    sigma=1.,
    maximise=True,
    seed=None,
):
    """ Run a genetic algorithm (GA) under the presented constraints, giving a
    population of artificial datasets for which the fitness function performs
    well.

    Parameters
    ----------
    fitness : func
        Any real-valued function that takes one `pandas.DataFrame` as argument.
        Use the `maximise` parameter to determine how `fitness` should be
        interpreted.
    size : int
        The size of the population to create.
    row_limits : list
        Lower and upper bounds on the number of rows a dataset can have.
    col_limits : list
        Lower and upper bounds on the number of columns a dataset can have.
    pdfs : list
        Used to create the initial population. These classes represent the
        distribution each column of a dataset can take. These distributions
        should take values either from the real numbers or the integers.

        By default, a random-parameter normal distribution is used. However,
        user-defined classes can be used so long as they have a `sample` method
        detailing how to sample from the distribution. For reproducibility,
        methods using NumPy are encouraged as the seed for the GA is set using
        `np.random.seed`.
    weights : list
        A probability distribution on how to select columns from
        `pdfs`. If `None`, pdfs will be chosen uniformly.
    stop : func
        A function which acts as a stopping condition on the fitness of the
        current population. If `None`, the GA will run to its maximum number of
        iterations.
    max_iter : int
        The maximum number of iterations to be carried out before terminating.
    best_prop : float
        The proportion of a population from which to select the "best" potential
        parents.
    lucky_prop : float
        The proportion of a population from which to sample some "lucky"
        potential parents.
    crossover_prob : float
        The probability with which to sample from the first parent over the
        second in a crossover operation.
    mutation_prob : float
        The probability of a particular "allele" in an individual being mutated.
    sigma : float
        When values are mutated in a dataset, they are resampled from the normal
        distribution centred around the current value with standard deviation
        `sigma`. By default this value is 1.
    maximise : bool
        Determines whether `fitness` is a function to be maximised or not.
        Fitness is maximised by default.
    seed : int
        The seed for a pseudo-random number generator for the run of the
        algorithm. If `None`, no seed is set.

    Returns
    -------
    population : list
        The final population.
    pop_fitness : array
        The fitness of `population`.
    all_populations : list
        Every individual in each generation.
    all_fitnesses : list
        Every individual's fitness in each generation.
    """

    if isinstance(seed, int):
        np.random.seed(seed)

    population = create_initial_population(
        size, row_limits, col_limits, pdfs, weights
    )

    pop_fitness = get_fitness(fitness, population)

    if stop:
        converged = stop(pop_fitness)
    else:
        converged = False

    itr = 0
    all_populations, all_fitnesses = [population], [pop_fitness]
    while itr < max_iter and not converged:

        parents = select_parents(
            population, pop_fitness, best_prop, lucky_prop, maximise
        )

        population = create_offspring(
            parents,
            size,
            crossover_prob,
            mutation_prob,
            row_limits,
            col_limits,
            pdfs,
            weights,
            sigma,
        )

        pop_fitness = get_fitness(fitness, population)

        all_populations.append(population)
        all_fitnesses.append(pop_fitness)

        if stop:
            converged = stop(pop_fitness)
        itr += 1

    return population, pop_fitness, all_populations, all_fitnesses
