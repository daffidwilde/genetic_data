""" Parameters for hypothesis testing, etc. """

from hypothesis import given
from hypothesis.strategies import floats, integers, tuples

size = integers(min_value=1, max_value=100)
prob = floats(min_value=0, max_value=1)

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

selection_limits = given(size=size,
                         row_limits=shapes,
                         col_limits=shapes,
                         weights=weights,
                         props=props)

offspring_limits = given(size=size,
                         row_limits=shapes,
                         col_limits=shapes,
                         weights=weights,
                         props=props,
                         prob=prob)

mutation_limits = given(size=size,
                        row_limits=shapes,
                        col_limits=shapes,
                        weights=weights,
                        mutation_prob=prob,
                        allele_prob=prob)

operator_limits = given(row_limits=shapes,
                        col_limits=shapes,
                        weights=weights,
                        prob=prob)
