""" Test(s) for the calculating of population fitness. """

from edo.fitness import get_fitness, get_population_fitness
from edo.pdfs import Normal, Poisson, Uniform
from edo.individual import create_individual
from edo.population import create_initial_population

from .util.parameters import INTEGER_INDIVIDUAL, POPULATION
from .util.trivials import trivial_fitness


@INTEGER_INDIVIDUAL
def test_get_fitness(row_limits, col_limits, weights):
    """ Create an individual and get its fitness. Then verify that the fitness
    is of the correct data type and has been added to the cache. """

    cache = {}
    families = [Normal, Poisson, Uniform]
    individual = create_individual(row_limits, col_limits, families, weights)
    dataframe = individual.dataframe

    fit = get_fitness(dataframe, trivial_fitness, cache)
    assert repr(dataframe) in cache
    assert isinstance(fit, float)


@INTEGER_INDIVIDUAL
def test_get_fitness_kwargs(row_limits, col_limits, weights):
    """ Create an individual and get its fitness with keyword arguments. Then
    verify that the fitness is of the correct data type and has been added to
    the cache. """

    cache = {}
    fitness_kwargs = {"arg": None}
    families = [Normal, Poisson, Uniform]
    individual = create_individual(row_limits, col_limits, families, weights)
    dataframe = individual.dataframe

    fit = get_fitness(dataframe, trivial_fitness, cache, fitness_kwargs)
    assert repr(dataframe) in cache
    assert isinstance(fit, float)


@POPULATION
def test_get_population_fitness(size, row_limits, col_limits, weights):
    """ Create a population and find its fitness. Verify that the fitness array
    is of the correct data type and size, and that they have each been added to
    the cache. """

    cache = {}
    families = [Normal, Poisson, Uniform]
    population = create_initial_population(size, row_limits, col_limits,
            families, weights)

    pop_fit = get_population_fitness(population, trivial_fitness, cache)
    assert len(pop_fit) == size
    for ind, fit in zip(population, pop_fit):
        assert repr(ind.dataframe) in cache
        assert isinstance(fit, float)
