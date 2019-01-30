""" .. The main script containing a generic genetic algorithm. """

import numpy as np

from .compact import compact_search_space
from .fitness import get_fitness
from .operators import selection
from .population import create_initial_population, create_new_population


def run_algorithm(
    fitness,
    size,
    row_limits,
    col_limits,
    pdfs,
    weights=None,
    stop=None,
    dwindle=None,
    max_iter=100,
    best_prop=0.25,
    lucky_prop=0,
    crossover_prob=0.5,
    mutation_prob=0.01,
    compact=None,
    maximise=False,
    seed=None,
    fitness_kwargs=None,
):
    """ Run a genetic algorithm under the presented constraints, giving a
    population of artificial datasets for which the given fitness function
    performs well.

    Parameters
    ----------
    fitness : func
        Any real-valued function that takes one :class:`pandas.DataFrame` as
        argument. Any further arguments should be passed to
        :code:`fitness_kwargs`.
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
        should be manipulated in a dataset.

        .. note::
            For reproducibility, a user-defined class' :code:`sample` method
            should use NumPy for any random elements as the seed for the GA is
            set using :func:`np.random.seed`.
    weights : list
        A probability distribution on how to select columns from
        :code:`pdfs`. If :code:`None`, pdfs will be chosen uniformly.
    stop : func
        A function which acts as a stopping condition on the GA. Such functions
        should take only the fitness of the current population as argument, and
        should return a boolean variable. If :code:`None`, the GA will run up
        until its maximum number of iterations.
    dwindle : func
        A function which acts as a means of dwindling the mutation probability.
        Such functions should take the current mutation probability and the
        current iteration as argument, and should return a new mutation
        probability. If :code:`None`, the GA will run with a constant mutation
        probability.
    max_iter : int
        The maximum number of iterations to be carried out before terminating.
    best_prop : float
        The proportion of a population from which to select the "best" potential
        parents.
    lucky_prop : float
        The proportion of a population from which to sample some "lucky"
        potential parents. Set to zero as standard.
    crossover_prob : float
        The probability with which to sample dimensions from the first parent
        over the second in a crossover operation. Defaults to 0.5.
    mutation_prob : float
        The probability of a particular characteristic in an individual's
        dataset being mutated. If using :code:`dwindle`, this is an initial
        probability.
    compact : float
        The proportion of the maximum iterations to reduce and focus the
        mutation search space. Defaults to `None` but must be between 0 and 1
        (not inclusive).
    maximise : bool
        Determines whether :code:`fitness` is a function to be maximised or not.
        Fitness scores are minimised by default.
    seed : int
        The seed for a particular run of the genetic algorithm. If :code:`None`,
        no seed is set.
    fitness_kwargs : dict
        Any additional parameters that need to be passed to :code:`fitness` should
        be placed here as a dictionary or suitable mapping.

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

    if seed is not None:
        np.random.seed(seed)

    population = create_initial_population(
        size, row_limits, col_limits, pdfs, weights
    )

    pop_fitness = get_fitness(fitness, population, fitness_kwargs)

    converged = False
    if stop:
        converged = stop(pop_fitness)

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

        pop_fitness = get_fitness(fitness, population, fitness_kwargs)

        all_populations.append(population)
        all_fitnesses.append(pop_fitness)

        if stop:
            converged = stop(pop_fitness)
        if dwindle:
            mutation_prob = dwindle(mutation_prob, itr)
        if compact is not None:
            pdfs = compact_search_space(parents, pdfs, itr, max_iter, compact)
        itr += 1

    return population, pop_fitness, all_populations, all_fitnesses
