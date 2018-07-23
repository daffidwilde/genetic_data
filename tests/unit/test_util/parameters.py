""" Parameters for hypothesis testing, etc. """

import itertools as itr
import numpy as np

from hypothesis import given
from hypothesis.strategies import floats, integers, sampled_from, tuples

SEED = integers(min_value=0, max_value=5)
SIZE = integers(min_value=2, max_value=10)
PROB = floats(min_value=0, max_value=1)
SMALL_PROB = floats(min_value=0, max_value=1e-3)

SHAPES = (
    tuples(
        integers(min_value=1, max_value=50), integers(min_value=1, max_value=50)
    )
    .map(sorted)
    .filter(lambda x: x[0] <= x[1])
)

UNIT = np.linspace(0, 1, 101)

WEIGHTS = sampled_from(
    [dist for dist in itr.product(UNIT, repeat=3) if sum(dist) == 1.0]
)


INDIVIDUAL = given(row_limits=SHAPES, col_limits=SHAPES, weights=WEIGHTS)

POPULATION = given(
    size=SIZE, row_limits=SHAPES, col_limits=SHAPES, weights=WEIGHTS
)

IND_FITNESS = given(
    row_limits=SHAPES, col_limits=SHAPES, weights=WEIGHTS, max_seed=SEED
)

POP_FITNESS = given(
    size=SIZE,
    row_limits=SHAPES,
    col_limits=SHAPES,
    weights=WEIGHTS,
    max_seed=SEED,
)

SELECTION = given(
    size=SIZE,
    row_limits=SHAPES,
    col_limits=SHAPES,
    weights=WEIGHTS,
    props=tuples(PROB, PROB).filter(lambda x: x[0] > 0.5 or x[1] > 0.5),
    max_seed=SEED,
)

SMALL_PROPS = given(
    size=SIZE,
    row_limits=SHAPES,
    col_limits=SHAPES,
    weights=WEIGHTS,
    props=tuples(SMALL_PROB, SMALL_PROB),
    max_seed=SEED,
)

OFFSPRING = given(
    size=SIZE,
    row_limits=SHAPES,
    col_limits=SHAPES,
    weights=WEIGHTS,
    props=tuples(PROB, PROB).filter(lambda x: x[0] > 0.5 or x[1] > 0.5),
    prob=PROB,
    max_seed=SEED,
)

MUTATION = given(
    row_limits=SHAPES,
    col_limits=SHAPES,
    weights=WEIGHTS,
    prob=PROB,
    sigma=floats(min_value=0),
)

CROSSOVER = given(
    row_limits=SHAPES, col_limits=SHAPES, weights=WEIGHTS, prob=PROB
)
