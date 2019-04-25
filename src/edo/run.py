""" .. The main script containing the evolutionary dataset algorithm. """

from collections import defaultdict
from glob import iglob
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
import yaml

import edo
from edo.fitness import get_population_fitness
from edo.individual import Individual
from edo.operators import selection, shrink
from edo.population import create_initial_population, create_new_population
from edo.write import write_generation


def run_algorithm(
    fitness,
    size,
    row_limits,
    col_limits,
    families,
    weights=None,
    stop=None,
    dwindle=None,
    max_iter=100,
    best_prop=0.25,
    lucky_prop=0,
    crossover_prob=0.5,
    mutation_prob=0.01,
    shrinkage=None,
    maximise=False,
    seed=None,
    processes=None,
    root=None,
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
        there can be of each type in :code:`families`.
    families : list
        Used to create the initial population and instruct the GA how a column
        should be manipulated in a dataset.

        .. note::
            For reproducibility, a user-defined class' :code:`sample` method
            should use NumPy for)any random elements as the seed for the GA is
            set using :func:`np.random.seed`.
    weights : list
        A probability distribution on how to select columns from
        :code:`families`. If :code:`None`, families will be chosen uniformly.
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
    shrinkage : float
        The relative size to shrink each parameter's limits by for each
        distribution in :code:`families`. Defaults to `None` but must be between 0
        and 1 (not inclusive).
    maximise : bool
        Determines whether :code:`fitness` is a function to be maximised or not.
        Fitness scores are minimised by default.
    seed : int
        The seed for a particular run of the genetic algorithm. If :code:`None`,
        no seed is set.
    processes : int
        The number of processes to use in order to parallelise several
        processes. Defaults to `None` where the algorithm is executed serially.
    root : str
        The directory in which to write all generations to file. Defaults to
        `None` where nothing is written to file. Instead, everything is kept in
        memory and returned at the end. If writing to file, one generation is
        held in memory at a time and everything is returned in `dask` objects.
    fitness_kwargs : dict
        Any additional parameters that need to be passed to :code:`fitness`
        should be placed here as a dictionary or suitable mapping.

    Returns
    -------
    population : list
        The final population.
    pop_fitness : list
        The fitness of all individuals in the final population.
    pop_history : list
        Every population in each generation.
    fit_history : list
        Every individual's fitness in each generation.
    """

    if seed is not None:
        np.random.seed(seed)

    population, pop_fitness = _initialise_algorithm(
        fitness,
        size,
        row_limits,
        col_limits,
        families,
        weights,
        processes,
        fitness_kwargs,
    )

    itr, converged = 0, False
    if stop:
        converged = stop(pop_fitness)

    if root is None:
        pop_history = _update_pop_history(population)
        fit_history = _update_fit_history(pop_fitness, itr)
    else:
        write_generation(population, pop_fitness, itr, root, processes)

    while itr < max_iter and not converged:

        itr += 1
        parents = selection(
            population, pop_fitness, best_prop, lucky_prop, maximise
        )
        families = _update_subtypes(parents, families)

        population = create_new_population(
            parents,
            size,
            crossover_prob,
            mutation_prob,
            row_limits,
            col_limits,
            families,
            weights,
        )

        pop_fitness = get_population_fitness(
            population, fitness, processes, fitness_kwargs
        )

        if root is None:
            pop_history = _update_pop_history(population, pop_history)
            fit_history = _update_fit_history(pop_fitness, itr, fit_history)
        else:
            write_generation(population, pop_fitness, itr, root, processes)

        if stop:
            converged = stop(pop_fitness)
        if dwindle:
            mutation_prob = dwindle(mutation_prob, itr)
        if shrinkage is not None:
            families = shrink(parents, families, itr, shrinkage)

    if root is not None:
        pop_history = _get_pop_history(root, itr)
        fit_history = _get_fit_history(root)

    return pop_history, fit_history


def _initialise_algorithm(
    fitness,
    size,
    row_limits,
    col_limits,
    families,
    weights,
    processes,
    fitness_kwargs=None,
):
    """ Initialise the algorithm: reset families and the fitness cache, generate
    an initial population and evaluate its fitness. """

    for family in families:
        family.reset()

    edo.cache.clear()

    population = create_initial_population(
        size, row_limits, col_limits, families, weights
    )
    pop_fitness = get_population_fitness(
        population, fitness, processes, fitness_kwargs
    )

    return population, pop_fitness


def _get_metadata_dicts(individual):
    """ Extract the dictionary form of  """

    meta_dicts = [pdf.to_dict() for pdf in individual.metadata]
    return meta_dicts


def _update_pop_history(population, pop_history=None):
    """ Add the current generation to the history. """

    population_dict = []
    for individual in population:
        meta_dicts = _get_metadata_dicts(individual)
        population_dict.append(Individual(individual.dataframe, meta_dicts))

    if pop_history is None:
        pop_history = [population_dict]
    else:
        pop_history.append(population_dict)

    return pop_history


def _update_fit_history(pop_fitness, itr, fit_history=None):
    """ Add the current generation's population fitness to the history. """

    size = len(pop_fitness)
    fitness_df = pd.DataFrame(
        {"fitness": pop_fitness, "generation": itr, "individual": range(size)}
    )

    if fit_history is None:
        fit_history = fitness_df
    else:
        fit_history = fit_history.append(fitness_df, ignore_index=True)

    return fit_history


def _update_subtypes(parents, families):
    """ Update the recorded subtypes for each pdf to be only those present in
    the parents. """

    subtypes = defaultdict(set)
    for parent in parents:
        for column in parent.metadata:
            subtypes[column.family].add(column.__class__)

    for pdf in families:
        pdf.subtypes = list(subtypes[pdf])

    return families


def _get_pop_history(root, itr):
    """ Read in the individuals from each generation. The dataset is given as a
    `dask.dataframe.core.DataFrame` and the metadata as a `dask.bag.core.Bag` of
    dictionaries. However, the individual is still an `Individual`. """

    pop_history = []
    for gen in range(itr):

        generation = []
        gen_path = Path(f"{root}/{gen}")
        for ind_dir in sorted(
            iglob(f"{gen_path}/*"), key=lambda path: int(path.split("/")[-1])
        ):
            ind_dir = Path(ind_dir)
            dataframe = dd.read_csv(ind_dir / "main.csv")
            with open(ind_dir / "main.meta", "r") as meta_file:
                metadata = yaml.load(meta_file, Loader=yaml.FullLoader)

            generation.append(Individual(dataframe, metadata))

        pop_history.append(generation)

    return pop_history


def _get_fit_history(root):
    """ Read in the fitness history from each generation in a run  as a
    `dask.dataframe.core.DataFrame`. """

    return dd.read_csv(f"{root}/fitness.csv")
