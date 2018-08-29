.. _mutate:

Mutation
========

To maintain a level of variety in a population and to force a GA to explore more
of the search space, new individuals are mutated immediately after the crossover
process.

There are many ways of mutating an individual. The most common is to
take each individual in the population of offspring and run along their alleles,
deciding whether or not to mutate that allele according to a small probability.
However, the method of mutation in Edo is not quite as simple. An individual is
mutated in the following way:

1. Mutate the number of rows and columns by adding and/or removing a line
   from each axis with equal probability. Lines are removed at random. Rows are
   added by sampling a new value from each column distribution and adding this
   vector to the bottom of the dataset. Columns are added in the same way as the 
   :ref:`creation <create-ind>` process. Note that the number of rows and
   columns will not mutate beyond the bounds passed to the genetic algorithm.

2. Then, with the dimensions of the dataset mutated, each value in the dataset
   is mutated using the same mutation probability. A value is mutated by
   replacing it with a single value sampled from the distribution associated
   with its column.

Parameters
----------

.. automodule:: edo.operators
   :members: mutation

Example
-------

Import :mod:`numpy` and the relevant pieces from :code:`edo`::

    >>> import numpy as np
    >>> from edo.individual import create_individual
    >>> from edo.operators import mutation
    >>> from edo.pdfs import Poisson

Set a seed::

    >>> np.random.seed(0)

Define the constraints and parameters of the simulation::
   
    >>> row_limits, col_limits = [1, 3], [1, 5]
    >>> pdfs = [Poisson]

Generate an individual::

    >>> individual = create_individual(row_limits, col_limits, pdfs)

This individual has this dataset:

.. csv-table:: The individual
   :file: individual.csv
   :align: center

.. include:: individual.rst

Set the mutation probability. This is deliberately large to make for a
substantial mutation::

    >>> mutation_prob = 0.5

Mutate the individual that was just created::

    >>> mutant = mutation(
    ...     individual, mutation_prob, row_limits, col_limits, pdfs
    ... )

This gives the following mutated dataset:

.. csv-table:: The mutant
   :file: mutant.csv
   :align: center

.. include:: mutant.rst
