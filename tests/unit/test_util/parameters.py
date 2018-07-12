""" Parameters for hypothesis testing, etc. """

from hypothesis import given
from hypothesis.strategies import floats, integers, tuples

size = integers(min_value=1, max_value=100)
max_seed = integers(min_value=1, max_value=20)
rate = floats(min_value=0, max_value=1)

props = tuples(floats(min_value=1e-10, max_value=1),
               floats(min_value=1e-10, max_value=1)) \
        .map(sorted).filter(lambda x: sum(x) <= 1.0)

shapes = tuples(integers(min_value=1, max_value=1e3),
                integers(min_value=1, max_value=1e3)) \
         .map(sorted).filter(lambda x: x[0] <= x[1])

weights = tuples(floats(min_value=1e-10, max_value=1),
                 floats(min_value=1e-10, max_value=1)) \
          .map(sorted).filter(lambda x: sum(x) <= 1.0)


individual_limits = given(row_limits=shapes,
                          col_limits=shapes,
                          weights=weights)

population_limits = given(size=size,
                          row_limits=shapes,
                          col_limits=shapes,
                          weights=weights)

ind_fitness_limits = given(row_limits=shapes,
                           col_limits=shapes,
                           weights=weights,
                           max_seed=max_seed)

pop_fitness_limits = given(size=size,
                           row_limits=shapes,
                           col_limits=shapes,
                           weights=weights,
                           max_seed=max_seed)

selection_limits = given(size=size,
                         row_limits=shapes,
                         col_limits=shapes,
                         weights=weights,
                         props=props,
                         max_seed=max_seed)

offspring_limits = given(size=size,
                         row_limits=shapes,
                         col_limits=shapes,
                         weights=weights,
                         props=props,
                         prob=rate,
                         max_seed=max_seed)

mutation_limits = given(size=size,
                        row_limits=shapes,
                        col_limits=shapes,
                        weights=weights,
                        mutation_rate=rate,
                        allele_prob=rate)

operator_limits = given(row_limits=shapes,
                        col_limits=shapes,
                        weights=weights,
                        prob=rate)
