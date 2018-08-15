""" Parameters for hypothesis testing, etc. """

import itertools as itr
import numpy as np

from hypothesis import given
from hypothesis.strategies import (
    floats,
    integers,
    sampled_from,
    tuples,
)

SIZE = integers(min_value=2, max_value=10)

INTS = integers(min_value=1, max_value=5)
SHAPES = tuples(INTS, INTS).map(sorted).filter(lambda x: x[0] <= x[1])

PROB = floats(min_value=0, max_value=1)

UNIT = np.linspace(0.01, 1, 100)
WEIGHTS = sampled_from(
    [dist for dist in itr.product(UNIT, repeat=3) if sum(dist) == 1.0]
)
