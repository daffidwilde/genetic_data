""" Tests for the shrinking of the search space. """

from edo import Family
from edo.distributions import Gamma, Normal, Poisson
from edo.fitness import get_population_fitness
from edo.operators import selection, shrink
from edo.population import create_initial_population

from .util.parameters import SHRINK
from .util.trivials import trivial_fitness


@SHRINK
def test_shrink(
    size, row_limits, col_limits, weights, props, maximise, compact_ratio, itr
):
    """ Test that the search space (the space of pdf parameter limits) of a
    hypothetical GA is reduced and centred around the best individuals'
    parameters at a particular iteration. """

    best_prop, lucky_prop = props
    distributions = [Gamma, Normal, Poisson]
    families = [Family(dist) for dist in distributions]

    population = create_initial_population(
        size, row_limits, col_limits, families, weights
    )

    pop_fitness = get_population_fitness(population, trivial_fitness)
    parents = selection(
        population, pop_fitness, best_prop, lucky_prop, maximise
    )

    families = shrink(parents, families, itr, compact_ratio)

    for family in families:
        assert family.distribution.param_limits.keys() == vars(family.make_instance()).keys()
