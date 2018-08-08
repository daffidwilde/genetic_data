""" The main script containing a generic genetic algorithm. """

import numpy as np

from genetic_data.pdfs import Normal
from genetic_data.operators import get_fitness, selection
from genetic_data.creation import (
    create_initial_population,
    create_new_population,
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
    maximise=False,
    seed=None,
):
    """ Run a genetic algorithm (GA) under the presented constraints, giving a
    population of artificial datasets for which the fitness function performs
    well.

    Parameters
    ----------
    fitness : func
        Any real-valued function that takes one :code:`pandas.DataFrame` as
        argument. Use the :code:`maximise` parameter to determine how
        :code:`fitness` should be interpreted.
    size : int
        The size of the population to create.
    row_limits : list
        Lower and upper bounds on the number of rows a dataset can have.
    col_limits : list
        Lower and upper bounds on the number of columns a dataset can have.
        Tuples can also be used to specify the min/maximum number of columns
        there can be of each type in :code:`pdfs`.
    pdfs : list
        Used to create the initial population and instruct the GA how a column
        should be manipulated in a dataset. These classes represent the
        distribution each column of a dataset can take. By default, a
        random-parameter normal distribution is used. For reproducibility, a
        user-defined class' :code:`sample` method should NumPy as the seed for
        the GA is set using :code:`np.random.seed`.
    weights : list
        A probability distribution on how to select columns from
        :code:`pdfs`. If :code:`None`, pdfs will be chosen uniformly.
    stop : func
        A function which acts as a stopping condition on the GA. Such functions
        should take only the fitness of the current population as argument.
        If :code:`None`, the GA will run up until its maximum number of
        iterations.
    max_iter : int
        The maximum number of iterations to be carried out before terminating.
    best_prop : float
        The proportion of a population from which to select the "best" potential
        parents.
    lucky_prop : float
        The proportion of a population from which to sample some "lucky"
        potential parents. Set to zero as standard.
    crossover_prob : float
        The probability with which to sample from the first parent over the
        second in a crossover operation.
    mutation_prob : float
        The probability of a particular characteristic in an individual's
        dataset being mutated.
    maximise : bool
        Determines whether :code:`fitness` is a function to be maximised or not.
        Fitness scores are minimised by default.
    seed : int
        The seed for a particular run of the genetic algorithm. If :code:`None`,
        no seed is set.

    Returns
    -------
    population : list
        The final population.
    pop_fitness : list
        The fitness of all individuals in the final population.
    all_populations : list
        Every population in each generation.
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

        parents = selection(
            population, pop_fitness, best_prop, lucky_prop, maximise
        )

        population = create_new_population(
            parents,
            size,
            crossover_prob,
            mutation_prob,
            row_limits,
            col_limits,
            pdfs,
            weights,
        )

        pop_fitness = get_fitness(fitness, population)

        all_populations.append(population)
        all_fitnesses.append(pop_fitness)

        if stop:
            converged = stop(pop_fitness)
        itr += 1

    return population, pop_fitness, all_populations, all_fitnesses
