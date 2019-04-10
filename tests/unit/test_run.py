""" Tests for the helper functions in the running of the EDO algorithm. """

from edo.individual import Individual
from edo.pdfs import Normal, Poisson, Uniform
from edo.population import create_initial_population
from edo.run import _initialise_algorithm, _update_subtypes

from .util.parameters import POPULATION
from .util.trivials import trivial_fitness


@POPULATION
def test_initialise_algorithm(size, row_limits, col_limits, weights):
    """ Test that the algorithm can be initialised correctly with a population
    and its fitness. """

    families = [Normal, Poisson, Uniform]
    fitness_kwargs = {"arg": None}

    population, pop_fitness = _initialise_algorithm(
        trivial_fitness, size, row_limits, col_limits, families, weights, None,
        fitness_kwargs
    )

    assert isinstance(population, list)
    assert len(population) == len(pop_fitness) == size

    for individual, fitness in zip(population, pop_fitness):
        assert isinstance(individual, Individual)
        assert isinstance(fitness, float)


@POPULATION
def test_update_subtypes(size, row_limits, col_limits, weights):
    """ Test that the subtypes of the present distributions can be updated. """

    families = [Normal, Poisson, Uniform]

    population = create_initial_population(
        size, row_limits, col_limits, families, weights
    )

    parents = population[:max(int(size / 5), 1)]
    parent_subtypes = {
        pdf.__class__ for parent in parents for pdf in parent.metadata
    }

    families = _update_subtypes(parents, families)
    updated_subtypes = {sub for family in families for sub in family.subtypes}

    assert parent_subtypes == updated_subtypes
