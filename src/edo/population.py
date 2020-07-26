""" Functions for the creation and updating of a population. """

from .individual import create_individual
from .operators import crossover, mutation


def create_initial_population(
    row_limits, col_limits, families, weights, random_states
):
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
    families : list
        A list of ``edo.Family`` instances that handle the column distribution
        classes.
    weights : list
        Relative weights with which to sample from ``families``. If ``None``,
        sampling is done uniformly.
    random_states : dict
        A mapping of the index of the population to a
        ``numpy.random.RandomState`` instance that is to be assigned to the
        individual at that index in the population.

    Returns
    -------
    population : list
        A population of newly created individuals.
    """

    population = [
        create_individual(row_limits, col_limits, families, weights, state)
        for _, state in random_states.items()
    ]

    return population


def create_new_population(
    parents,
    population,
    crossover_prob,
    mutation_prob,
    row_limits,
    col_limits,
    families,
    weights,
    random_states,
):
    """ Given a set of potential parents to be carried into the next generation,
    create offspring from pairs within that set until there are enough
    individuals.

    Parameters
    ----------
    parents : list
        A list of `edo.individual.Individual` instances used to create new
        offspring.
    population : list
        The current population.
    crossover_prob : float
        The probability with which to sample dimensions from the first parent
        over the second during crossover.
    mutation_prob : float
        The probability with which to mutate a component of a newly created
        individual.
    row_limits : list
        Limits on the number of rows a dataset can have.
    col_limits : list
        Limits on the number of columns a dataset can have.
    families : list
        The ``edo.Family`` instances from which to draw distribution instances.
    weights : list
        Weights used to sample elements from ``families``.
    random_states : dict
        The PRNGs assigned to each individual in the population.
    """

    parent_idxs = [population.index(parent) for parent in parents]
    available_states = [
        state for i, state in random_states.items() if i not in parent_idxs
    ]

    new_population = parents
    for state in available_states:
        parent1_idx, parent2_idx = state.choice(len(parents), size=2)
        parents_ = parents[parent1_idx], parents[parent2_idx]
        offspring = crossover(
            *parents_, col_limits, families, state, crossover_prob
        )
        mutant = mutation(
            offspring, mutation_prob, row_limits, col_limits, families, weights
        )
        for col, meta in zip(*mutant):
            mutant.dataframe[col] = mutant.dataframe[col].astype(meta.dtype)
        new_population.append(mutant)

    return new_population
