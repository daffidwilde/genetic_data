.. _cross:

Crossover
=========

A crossover operator defines how two individuals should be combined to create a
new individual (or individuals). Importantly, the crossover operator allows for
the preservation of preferable characteristics found by the genetic algorithm.

In EDO, the crossover operator returns exactly one individual from a pair of
parents. As is discussed elsewhere_, an individual is created by sampling its
dimensions and then its values. Creating an offspring is done in the same way
except it inherits these characteristics from its parents:

1. Inherit a number of rows and a number of columns from either parent,
   independently and uniformly. This is the skeleton of the dataset.
2. Pool together the columns (and respective column distributions) from both
   parents.
3. Sample from this pool uniformly (and without replacement) to fill in 
   the columns of the offspring's dataset. Now the dataset has values and
   instructions on how to manipulate it.
4. Remove surplus rows as required, and fill in any missing values using the
   corresponding column information. This is now a complete individual.

Before this offspring is added to population, it must undergo mutation_.

Example
-------

Consider the following example where two individuals are created::

    >>> import numpy as np
    >>> from edo import Family
    >>> from edo.distributions import Poisson
    >>> from edo.individual import create_individual
    >>> from edo.operators import crossover
    >>> 
    >>> row_limits, col_limits = [1, 3], [2, 3]
    >>> families = [Family(Poisson)]
    >>> states = [np.random.RandomState(i) for i in range(2)]
    >>> 
    >>> parents = [
    ...     create_individual(
    ...         row_limits, col_limits, families, weights=None, random_state=state
    ...     ) for state in states
    ... ]

These individuals' dataframes look like this::

    >>> parents[0].dataframe
        0  1   2
    0  12  0  12
    >>> parents[1].dataframe
       0  1  2
    0  0  5  7
    1  4  4  9

And their metadata like this::

    >>> parents[0].metadata
    [Poisson(lam=7.15), Poisson(lam=0.87), Poisson(lam=8.33)]
    >>> parents[1].metadata
    [Poisson(lam=7.2), Poisson(lam=3.97), Poisson(lam=8.01)]

Now, we create a PRNG for the offspring and apply the crossover::

    >>> state = np.random.RandomState(2)
    >>> offspring = crossover(*parents, col_limits, families, state)
    >>> 
    >>> offspring.dataframe
       0   1  2
    0  0  12  7
    >>> offspring.metadata
    [Poisson(lam=7.2), Poisson(lam=8.33), Poisson(lam=8.01)]

.. _mutation: mutation.rst
