""" .. The main script containing the evolutionary dataset algorithm. """

from collections import defaultdict
from glob import iglob
from pathlib import Path

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import yaml

import edo
from edo.fitness import get_population_fitness
from edo.individual import Individual
from edo.operators import selection, shrink
from edo.population import create_initial_population, create_new_population
from edo.write import write_fitness, write_individual


class DataOptimiser:
    """ The (evolutionary) dataset optimiser. A class for generating data for a
    given fitness function and evolutionary parameters.

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
        Used to create the initial population and instruct the EA how a column
        should be manipulated in a dataset.

        .. note::
            For reproducibility, a user-defined class' :code:`sample` method
            should use NumPy for any random elements as the seed for the EA is
            set using :func:`np.random.seed`.
    weights : list
        A probability distribution on how to select columns from
        :code:`families`. If :code:`None`, families will be chosen uniformly.
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
        distribution in :code:`families`. Defaults to `None` but must be between
        0 and 1 (not inclusive).
    maximise : bool
        Determines whether :code:`fitness` is a function to be maximised or not.
        Fitness scores are minimised by default.
    """

    def __init__(
        self,
        fitness,
        size,
        row_limits,
        col_limits,
        families,
        weights=None,
        max_iter=100,
        best_prop=0.25,
        lucky_prop=0,
        crossover_prob=0.5,
        mutation_prob=0.01,
        shrinkage=None,
        maximise=False,
    ):

        edo.cache.clear()
        for family in families:
            family.reset()

        self.fitness = fitness
        self.size = size
        self.row_limits = row_limits
        self.col_limits = col_limits
        self.families = families
        self.weights = weights
        self.max_iter = max_iter
        self.best_prop = best_prop
        self.lucky_prop = lucky_prop
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.shrinkage = shrinkage
        self.maximise = maximise

        self.converged = False
        self.generation = 0
        self.population = None
        self.pop_fitness = None
        self.pop_history = None
        self.fit_history = None

    def stop(self, **kwargs):
        """ A placeholder for a function which acts as a stopping condition on
        the EA. """

    def dwindle(self, **kwargs):
        """ A placeholder for a function which can adjust (typically, reduce)
        the mutation probability over the run of the EA. """

    def run(self, root=None, processes=None, seed=None, kwargs=None):
        """ Run the evolutionary algorithm under the given constraints.

        Parameters
        ----------
        seed : int
            The random seed for a particular run of the algorithm. If
            :code:`None`, no seed is set.
        processes : int
            The number of processes to use in order to parallelise several
            stages of the EA. Defaults to `None` where the algorithm is executed
            serially.
        root : str
            The directory in which to write all generations to file. Defaults to
            `None` where nothing is written to file. Instead, everything is kept
            in memory and returned at the end. If writing to file, one
            generation is held in memory at a time and everything is returned in
            `dask` objects.
        kwargs : dict
            Any additional parameters that need to be passed to the functions
            for fitness, stopping or dwindling should be placed here as a
            dictionary or suitable mapping.

        Returns
        -------
        pop_history : list
            Every individual in each generation as a nested list of `Individual`
            instances.
        fit_history : `pd.DataFrame` or `dask.core.dataframe.DataFrame`
            Every individual's fitness in each generation.
        """

        if seed is not None:
            np.random.seed(seed)

        self._initialise_run(processes, kwargs)
        self._update_histories(root, processes)
        self.stop(**kwargs)
        while self.generation < self.max_iter and not self.converged:

            self.generation += 1
            self._get_next_generation(processes, kwargs)
            self._update_histories(root, processes)
            self.stop(**kwargs)
            self.dwindle(**kwargs)

        if root is not None:
            self.pop_history = _get_pop_history(root, self.generation)
            self.fit_history = _get_fit_history(root)

        return self.pop_history, self.fit_history

    def _initialise_run(self, processes, kwargs):
        """ Create the initial population and get its fitness. """

        self.population = create_initial_population(
            self.size,
            self.row_limits,
            self.col_limits,
            self.families,
            self.weights,
        )
        self.pop_fitness = get_population_fitness(
            self.population, self.fitness, processes, kwargs
        )

    def _get_next_generation(self, processes, kwargs):
        """ Create the next population, update the family subtypes and get the
        new population's fitness. """

        parents = selection(
            self.population,
            self.pop_fitness,
            self.best_prop,
            self.lucky_prop,
            self.maximise,
        )

        self._update_subtypes(parents)

        self.population = create_new_population(
            parents,
            self.size,
            self.crossover_prob,
            self.mutation_prob,
            self.row_limits,
            self.col_limits,
            self.families,
            self.weights,
        )

        self.pop_fitness = get_population_fitness(
            self.population, self.fitness, processes, kwargs
        )

        self.families = shrink(
            parents, self.families, self.generation, self.shrinkage
        )

    def _update_pop_history(self):
        """ Add the current generation to the history. """

        population_with_dicts = []
        for individual in self.population:
            meta_dicts = _get_metadata_dicts(individual)
            population_with_dicts.append(
                Individual(individual.dataframe, meta_dicts)
            )

        if self.pop_history is None:
            self.pop_history = [population_with_dicts]
        else:
            self.pop_history.append(population_with_dicts)

    def _update_fit_history(self):
        """ Add the current generation's population fitness to the history. """

        fitness_df = pd.DataFrame(
            {
                "fitness": self.pop_fitness,
                "generation": self.generation,
                "individual": range(self.size),
            }
        )

        if self.fit_history is None:
            self.fit_history = fitness_df
        else:
            self.fit_history = self.fit_history.append(
                fitness_df, ignore_index=True
            )

    def _write_generation(self, root, processes):
        """ Write all individuals in a generation and their collective fitnesses
        to file at the generation's directory in `root`. """

        tasks = (
            *[
                write_individual(individual, self.generation, i, root)
                for i, individual in enumerate(self.population)
            ],
            write_fitness(self.pop_fitness, self.generation, root),
        )

        if processes is None:
            dask.compute(*tasks, scheduler="single-threaded")
        else:
            dask.compute(*tasks, num_workers=processes)

    def _update_histories(self, root, processes):
        """ Update the population and fitness histories. """

        if root is None:
            self._update_pop_history()
            self._update_fit_history()
        else:
            self._write_generation(root, processes)

    def _update_subtypes(self, parents):
        """ Update the recorded subtypes for each pdf to be only those present
        in the parents. """

        subtypes = defaultdict(set)
        for parent in parents:
            for column in parent.metadata:
                subtypes[column.family].add(column.__class__)

        for pdf in self.families:
            pdf.subtypes = list(subtypes[pdf])


def _get_pop_history(root, generation):
    """ Read in the individuals from each generation. The dataset is given
    as a `dask.dataframe.core.DataFrame` and the metadata as a list of
    dictionaries. However, the individual is still an `Individual`. """

    pop_history = []
    for gen in range(generation):

        population = []
        gen_path = Path(f"{root}/{gen}")
        for ind_dir in sorted(
            iglob(f"{gen_path}/*"), key=lambda path: int(path.split("/")[-1])
        ):
            ind_dir = Path(ind_dir)
            dataframe = dd.read_csv(ind_dir / "main.csv")
            with open(ind_dir / "main.meta", "r") as meta_file:
                metadata = yaml.load(meta_file, Loader=yaml.FullLoader)

            population.append(Individual(dataframe, metadata))

        pop_history.append(population)

    return pop_history


def _get_fit_history(root):
    """ Read in the fitness history from each generation in a run  as a
    `dask.dataframe.core.DataFrame`. """

    return dd.read_csv(f"{root}/fitness.csv")


def _get_metadata_dicts(individual):
    """ Extract the dictionary form of  """

    meta_dicts = [pdf.to_dict() for pdf in individual.metadata]
    return meta_dicts