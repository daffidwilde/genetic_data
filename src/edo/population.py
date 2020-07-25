""" A script containing functions for each of the components of the genetic
algorithm. """

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
        A list of Family instances that handle the column distribution classes.
        Each column distribution class must have a `.sample()` method.
    weights : list
        A sequence of relative weights the same length as `column_classes`. This
        acts as a loose probability distribution from which to sample column
        classes. If `None`, column classes are sampled equally.
    random_states : dict
        The `numpy.random.RandomState` instances to be assigned to the
        individuals in the population.

    Returns
    -------
    population : list
        A collection of individuals.
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
    create offspring from pairs withi} that set until there are enough
    individuals. Each individual offspring is formed using a crossover operator
    on the two parent individuals and then mutating them according to the
    probability `mutation_prob`. """

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
