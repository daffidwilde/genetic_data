""" The evolutionary dataset optimisation algorithm class. """

from collections import defaultdict
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd

from edo.fitness import get_population_fitness, write_fitness
from edo.individual import Individual
from edo.operators import selection, shrink
from edo.population import create_initial_population, create_new_population


class DataOptimiser:
    """ The (evolutionary) dataset optimiser. A class that generates data for a
    given fitness function and evolutionary parameters.

    Parameters
    ----------
    fitness : func
        Any real-valued function that at least takes an instance of
        ``Individual`` as argument. Any further arguments should be passed in
        the ``kwargs`` parameter of the ``run`` method.
    size : int
        The size of the population to create.
    row_limits : list
        Lower and upper bounds on the number of rows a dataset can have.
    col_limits : list
        Lower and upper bounds on the number of columns a dataset can have.

        Tuples can also be used to specify the min/maximum number of columns
        there can be of each element in ``families``.
    families : list
        A list of ``edo.Family`` instances that handle the distribution classes
        used to populate the individuals in the EA.
    weights : list
        A set of relative weights on how to select elements from ``families``.
        If ``None``, they will be chosen uniformly.
    max_iter : int
        The maximum number of iterations to be carried out before terminating.
    best_prop : float
        The proportion of a population from which to select the "best"
        individuals to be parents.
    lucky_prop : float
        The proportion of a population from which to sample some "lucky"
        individuals to be parents. Defaults to ``0``.
    crossover_prob : float
        The probability with which to sample dimensions from the first parent
        over the second in a crossover operation. Defaults to ``0.5``.
    mutation_prob : float
        The probability of a particular characteristic of an individual being
        mutated. If using a ``dwindle`` method, this is an initial probability.
    shrinkage : float
        The relative size to shrink each parameter's limits by for each
        distribution in ``families``. Defaults to ``None`` but must be between
        0 and 1 (exclusive).
    maximise : bool
        Determines whether ``fitness`` is a function to be maximised or not.
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

    def run(
        self,
        root=None,
        random_state=None,
        processes=None,
        fitness_kwargs=None,
        stop_kwargs=None,
        dwindle_kwargs=None,
    ):
        """ Run the evolutionary algorithm under the given constraints.

        Parameters
        ----------
        root : str, optional
            The directory in which to write all generations to file. If
            ``None``, nothing is written to file. Instead, every generation is
            kept in memory and is returned at the end. If writing to file, one
            generation is held in memory at a time and everything is returned
            upon termination as a tuple containing ``dask`` objects.
        random_state : int or np.ran.RandomState, optional
            The random seed or state for a particular run of the algorithm. If
            ``None``, the default PRNG is used.
        processes : int, optional
            The number of parallel processes to use when calculating the
            population fitness. If ``None`` then a single-thread scheduler is
            used.
        fitness_kwargs : dict, optional
            Any additional parameters for the fitness function should be placed
            here.
        stop_kwargs : dict, optional
            Any additional parameters for the ``stop`` method should be placed
            here.
        dwindle_kwargs : dict, optional
            Any additional parameters for the ``dwindle`` method should be
            placed here.

        Returns
        -------
        pop_history : list
            Every individual in each generation as a nested list of
            ``Individual`` instances.
        fit_history : ``pd.DataFrame`` or ``dask.dataframe.DataFrame``
            Every individual's fitness in each generation.
        """

        if fitness_kwargs is None:
            fitness_kwargs = {}
        if stop_kwargs is None:
            stop_kwargs = {}
        if dwindle_kwargs is None:
            dwindle_kwargs = {}

        if isinstance(random_state, int):
            self.random_state = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.mtrand._rand

        self._initialise_run(processes, **fitness_kwargs)
        self._update_histories(root)
        self.stop(**stop_kwargs)
        while self.generation < self.max_iter and not self.converged:

            self.generation += 1
            self._get_next_generation(processes, **fitness_kwargs)
            self._update_histories(root)
            self.stop(**stop_kwargs)
            self.dwindle(**dwindle_kwargs)

        if root is not None:
            distributions = [family.distribution for family in self.families]
            self.pop_history = _get_pop_history(
                root, self.generation, distributions
            )
            self.fit_history = _get_fit_history(root)

        return self.pop_history, self.fit_history

    def _initialise_run(self, processes, **fitness_kwargs):
        """ Create the initial population and get its fitness. """

        state_seeds = self.random_state.randint(
            np.iinfo(np.int32).max, size=self.size
        )
        self.states = {
            i: np.random.RandomState(seed) for i, seed in enumerate(state_seeds)
        }

        family_seeds = self.random_state.randint(
            np.iinfo(np.int32).max, size=len(self.families)
        )
        for family, seed in zip(self.families, family_seeds):
            family.random_state = np.random.RandomState(seed)

        self.population = create_initial_population(
            self.row_limits,
            self.col_limits,
            self.families,
            self.weights,
            self.states,
        )

        self.pop_fitness = get_population_fitness(
            self.population, self.fitness, processes, **fitness_kwargs
        )

    def _get_next_generation(self, processes, **kwargs):
        """ Create the next population via selection, crossover and mutation,
        update the family subtypes and get the new population's fitness. """

        parents = selection(
            self.population,
            self.pop_fitness,
            self.best_prop,
            self.lucky_prop,
            self.random_state,
            self.maximise,
        )

        self._update_subtypes(parents)

        self.population = create_new_population(
            parents,
            self.population,
            self.crossover_prob,
            self.mutation_prob,
            self.row_limits,
            self.col_limits,
            self.families,
            self.weights,
            self.states,
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
                individual_dir, distributions, root, method="dask"
            )

            population.append(individual)

        pop_history.append(population)

    return pop_history


def _get_fit_history(root):
    """ Read in the fitness history from each generation in a run  as a
    `dask.dataframe.core.DataFrame`. """

    return dd.read_csv(f"{root}/fitness.csv")
