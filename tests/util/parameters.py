""" Parameters for hypothesis testing, etc. """

import itertools as itr

import numpy as np
from hypothesis import given
from hypothesis.strategies import (
    booleans,
    floats,
    integers,
    sampled_from,
    tuples,
)

SIZE = integers(min_value=2, max_value=5)
INTS = integers(min_value=1, max_value=5)
PROB = floats(min_value=0, max_value=1)
SMALL_PROB = floats(min_value=0, max_value=1e-3)
TUPS = tuples(
    integers(min_value=0, max_value=2),
    integers(min_value=0, max_value=2),
    integers(min_value=0, max_value=2),
)

SHAPES = tuples(INTS, INTS).map(sorted).filter(lambda x: x[0] <= x[1])

INT_TUPS = tuples(INTS, TUPS).filter(
    lambda x: sum(x[1]) >= x[0] and sum(x[1]) > 0
)

TUP_INTS = tuples(TUPS, INTS).filter(
    lambda x: sum(x[0]) <= x[1] and sum(x[0]) > 0
)

TUPLES = tuples(TUPS, TUPS).filter(
    lambda x: sum(x[0]) <= sum(x[1])
    and sum(x[0]) > 0
    and sum(x[1]) > 0
    and x[0][0] <= x[1][0]
    and x[0][1] <= x[1][1]
    and x[0][2] <= x[1][2]
)

UNIT = np.linspace(0.01, 1, 100)

PROPS = tuples(PROB, PROB).filter(lambda x: x[0] > 0.5 or x[1] > 0.5)

WEIGHTS = sampled_from(
    [dist for dist in itr.product(UNIT, repeat=3) if sum(dist) == 1.0]
)


INTEGER_INDIVIDUAL = given(
    row_limits=SHAPES, col_limits=SHAPES, weights=WEIGHTS, seed=INTS
)

INTEGER_TUPLE_INDIVIDUAL = given(
    row_limits=SHAPES, col_limits=INT_TUPS, weights=WEIGHTS, seed=INTS
)

TUPLE_INTEGER_INDIVIDUAL = given(
    row_limits=SHAPES, col_limits=TUP_INTS, weights=WEIGHTS, seed=INTS
)

TUPLE_INDIVIDUAL = given(
    row_limits=SHAPES, col_limits=TUPLES, weights=WEIGHTS, seed=INTS
)

POPULATION = given(
    size=SIZE, row_limits=SHAPES, col_limits=SHAPES, weights=WEIGHTS
)

POP_FITNESS = given(
    size=SIZE,
    row_limits=SHAPES,
    col_limits=SHAPES,
    weights=WEIGHTS,
    processes=INTS,
)

INTEGER_CROSSOVER = given(
    row_limits=SHAPES, col_limits=SHAPES, weights=WEIGHTS, prob=PROB, seed=INTS
)

INTEGER_TUPLE_CROSSOVER = given(
    row_limits=SHAPES,
    col_limits=INT_TUPS,
    weights=WEIGHTS,
    prob=PROB,
    seed=INTS,
)

TUPLE_INTEGER_CROSSOVER = given(
    row_limits=SHAPES,
    col_limits=TUP_INTS,
    weights=WEIGHTS,
    prob=PROB,
    seed=INTS,
)

TUPLE_CROSSOVER = given(
    row_limits=SHAPES, col_limits=TUPLES, weights=WEIGHTS, prob=PROB, seed=INTS
)

INTEGER_MUTATION = given(
    row_limits=SHAPES, col_limits=SHAPES, weights=WEIGHTS, prob=PROB, seed=INTS
)

INTEGER_TUPLE_MUTATION = given(
    row_limits=SHAPES,
    col_limits=INT_TUPS,
    weights=WEIGHTS,
    prob=PROB,
    seed=INTS,
)

TUPLE_INTEGER_MUTATION = given(
    row_limits=SHAPES,
    col_limits=TUP_INTS,
    weights=WEIGHTS,
    prob=PROB,
    seed=INTS,
)

TUPLE_MUTATION = given(
    row_limits=SHAPES, col_limits=TUPLES, weights=WEIGHTS, prob=PROB, seed=INTS
)

OFFSPRING = given(
    size=SIZE,
    row_limits=SHAPES,
    col_limits=SHAPES,
    weights=WEIGHTS,
    props=PROPS,
    crossover_prob=PROB,
    mutation_prob=PROB,
    maximise=booleans(),
)

SELECTION = given(
    size=SIZE,
    row_limits=SHAPES,
    col_limits=SHAPES,
    weights=WEIGHTS,
    props=PROPS,
    maximise=booleans(),
)

SHRINK = given(
    size=SIZE,
    row_limits=SHAPES,
    col_limits=SHAPES,
    weights=WEIGHTS,
    props=PROPS,
    maximise=booleans(),
    compact_ratio=PROB,
    itr=INTS,
)

SMALL_PROPS = given(
    size=SIZE,
    row_limits=SHAPES,
    col_limits=SHAPES,
    weights=WEIGHTS,
    props=tuples(SMALL_PROB, SMALL_PROB),
    maximise=booleans(),
)
