""" .. The main script containing the evolutionary dataset algorithm. """

from collections import defaultdict
from glob import iglob
from pathlib import Path

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd

import edo
from edo.fitness import get_population_fitness, write_fitness
from edo.individual import Individual
from edo.operators import selection, shrink
from edo.population import create_initial_population, create_new_population


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
        A list of `Family` instances that handle the distribution classes used
        to populate the individuals in the EA.

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
        self.pop_history = []
        self.fit_history = pd.DataFrame()

    def stop(self, **kwargs):
        """ A placeholder for a function which acts as a stopping condition on
        the EA. """

    def dwindle(self, **kwargs):
        """ A placeholder for a function which can adjust (typically, reduce)
        the mutation probability over the run of the EA. """

    def run(self, root=None, seed=None, processes=None, kwargs=None):
        """ Run the evolutionary algorithm under the given constraints.

        Parameters
        ----------
        root : str, optional
            The directory in which to write all generations to file. Defaults to
            `None` where nothing is written to file. Instead, everything is kept
            in memory and returned at the end. If writing to file, one
            generation is held in memory at a time and everything is returned in
            `dask` objects.
        seed : int, optional
            The random seed for a particular run of the algorithm. If
            :code:`None`, no seed is set.
        processes : int, optional
            The number of parallel processes to use when calculating the
            population fitness. If `None` then a single-thread scheduler is
            used.
        kwargs : dict, optional
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

        if kwargs is None:
            kwargs = {}

        if seed is not None:
            np.random.seed(seed)

        self._initialise_run(processes, **kwargs)
        self._update_histories(root)
        self.stop(**kwargs)
        while self.generation < self.max_iter and not self.converged:

            self.generation += 1
            self._get_next_generation(processes, **kwargs)
            self._update_histories(root)
            self.stop(**kwargs)
            self.dwindle(**kwargs)

        if root is not None:
            distributions = [family.distribution for family in self.families]
            self.pop_history = _get_pop_history(
                root, self.generation, distributions
            )
            self.fit_history = _get_fit_history(root)

        return self.pop_history, self.fit_history

    def _initialise_run(self, processes, **kwargs):
        """ Create the initial population and get its fitness. """

        self.population = create_initial_population(
            self.size,
            self.row_limits,
            self.col_limits,
            self.families,
            self.weights,
        )
        self.pop_fitness = get_population_fitness(
            self.population, self.fitness, processes, **kwargs
        )

    def _get_next_generation(self, processes, **kwargs):
        """ Create the next population via selection, crossover and mutation,
        update the family subtypes and get the new population's fitness. """

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
            self.population, self.fitness, processes, **kwargs
        )

        if self.shrinkage is not None:
            self.families = shrink(
                parents, self.families, self.generation, self.shrinkage
            )

    def _update_pop_history(self):
        """ Add the current generation to the history. """

        self.pop_history.append(self.population)

    def _update_fit_history(self):
        """ Add the current generation's population fitness to the history. """

        fitness_df = pd.DataFrame(
            {
                "fitness": self.pop_fitness,
                "generation": self.generation,
                "individual": range(self.size),
            }
        )

        self.fit_history = self.fit_history.append(
            fitness_df, ignore_index=True
        )

    def _write_generation(self, root):
        """ Write all individuals in a generation and their collective fitnesses
        to file at the generation's directory in `root`. """

        write_fitness(self.pop_fitness, self.generation, root)
        for idx, individual in enumerate(self.population):
            individual.to_file(f"{root}/{self.generation}/{idx}/", root)

    def _update_histories(self, root):
        """ Update the population and fitness histories. """

        if root is None:
            self._update_pop_history()
            self._update_fit_history()
        else:
            self._write_generation(root)

    def _get_current_subtypes(self, parents):
        """ Get a dictionary mapping each family to all the subtype IDs that are
        present in the parents. """

        family_to_subtype_ids = defaultdict(list)
        for parent in parents:
            for pdf in parent.metadata:
                family = pdf.family
                subtype_id = pdf.subtype_id
                record_subtypes = family_to_subtype_ids[family]
                if subtype_id not in record_subtypes:
                    family_to_subtype_ids[family].append(subtype_id)

        return family_to_subtype_ids

    def _update_subtypes(self, parents):
        """ Update the current subtypes for each family to be those present in
        the parents. """

        current_subtypes = self._get_current_subtypes(parents)
        for family, current_ids in current_subtypes.items():
            family.subtypes = {
                subtype_id: family.all_subtypes[subtype_id]
                for subtype_id in current_ids
            }


def _get_pop_history(root, generation, distributions):
    """ Read in the individuals from each generation. The dataset is given
    as a `dask.dataframe.core.DataFrame` but the metadata are recovered
    instances of their original class subtypes. """

    pop_history = []
    for gen in range(generation):

        population = []
        gen_path = Path(f"{root}/{gen}")
        for ind_dir in sorted(
            gen_path.glob("*"), key=lambda path: int(path.stem)
        ):
            individual_dir = Path(ind_dir)
            individual = Individual.from_file(
                individual_dir, distributions, root, method=dd
            )

            population.append(individual)

        pop_history.append(population)

    return pop_history


def _get_fit_history(root):
    """ Read in the fitness history from each generation in a run  as a
    `dask.dataframe.core.DataFrame`. """

    return dd.read_csv(f"{root}/fitness.csv")
