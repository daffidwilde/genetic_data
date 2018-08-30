""" Test(s) for the calculating of population fitness. """

import numpy as np

from edo.population import create_initial_population
from edo.fitness import get_fitness
from edo.pdfs import Gamma, Normal, Poisson

from .util.trivials import trivial_fitness
from .util.parameters import FITNESS


@FITNESS
def test_get_fitness(size, row_limits, col_limits, weights):
    """ Create a population and get its fitness. Then verify that the
    fitness is of the correct size and data type. """

    pdfs = [Gamma, Normal, Poisson]
    population = create_initial_population(
        size, row_limits, col_limits, pdfs, weights
    )

    pop_fitness = get_fitness(trivial_fitness, population)

    assert len(pop_fitness) == size
    assert np.array(pop_fitness).dtype == "float"
