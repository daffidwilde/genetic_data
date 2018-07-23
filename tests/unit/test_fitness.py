""" Test the fitness calculation process. """

import numpy as np

from genetic_data.components import create_initial_population, get_fitness
from genetic_data.pdfs import Gamma, Normal, Poisson

from test_util.parameters import FITNESS
from test_util.trivials import trivial_fitness


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
