.. _mutate:

Mutation
========

To maintain a level of variety in a population and to force the evolutionary
algorithm to explore more of the search space, new individuals are mutated
immediately after their creation during the crossover process.

The mutation process in EDO is not quite as simple as in a traditional genetic
algorithm. This is due to the representation of individuals. An individual is
mutated in the following way:

1. Mutate the number of rows and columns by adding and/or removing a line
   from each axis with the same probability. Lines are removed at random. Rows
   are added by sampling a new value from each current column distribution and
   adding them to the bottom of the dataset. Columns are added in the same way
   as in the :ref:`creation process <create-ind>`. Note that the number of rows
   and columns will not mutate beyond the bounds passed in ``col_limits``.
2. With the dimensions of the dataset mutated, each value in the dataset is
   mutated using the same mutation probability. A value is mutated by replacing
   it with a single value sampled from the distribution associated with its
   column.

Example
-------

Consider the following mutation of an individual::

    >>> import numpy as np
    >>> from edo import Family
    >>> from edo.distributions import Poisson
    >>> from edo.individual import create_individual
    >>> from edo.operators import mutation
    >>> 
    >>> row_limits, col_limits = [3, 5], [2, 5]
    >>> families = [Family(Poisson)]
    >>> state = np.random.RandomState(0)
    >>> 
    >>> individual = create_individual(
    ...     row_limits, col_limits, families, weights=None, random_state=state
    ... )

The individual looks like this::

    >>> individual.dataframe
        0  1  2  3  4
    0  12  8  4  1  7
    1   6  6  5  1  5
    2   8  7  7  1  3
    >>> individual.metadata
    [Poisson(lam=7.15), Poisson(lam=7.74), Poisson(lam=6.53), Poisson(lam=2.83), Poisson(lam=6.92)]

Now we can mutate this individual after setting the mutation probability. This
is deliberately large to make for a substantial mutation::

    >>> mutation_prob = 0.7
    >>> mutant = mutation(individual, mutation_prob, row_limits, col_limits, families)

This gives the following individual::

    >>> mutant.dataframe
        0  1  2  3
    0   8  4  1  5
    1  11  3  4  5
    2   9  7  3  3
    >>> mutant.metadata
    [Poisson(lam=7.74), Poisson(lam=6.53), Poisson(lam=2.83), Poisson(lam=6.92)]
