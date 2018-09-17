.. _mutate:

Mutation
========

To maintain a level of variety in a population and to force the genetic
algorithm to explore more of the search space, new individuals are mutated
immediately after their creation during the crossover process.

There are many ways of mutating an individual but the most common is to take
each new offspring and run along their alleles, deciding whether or not to
mutate that allele according to a small probability. For the chromosome
representation of individuals, this means turning a 0 into a 1, and vice versa.

The method of mutation in Edo is not quite as simple. An individual is
mutated in the following way:

1. Mutate the number of rows and columns by adding and/or removing a line
   from each axis with the same probability. Lines are removed at random. Rows
   are added by sampling a new value from each current column distribution and
   adding them to the bottom of the dataset. Columns are added in the same way
   as in the :ref:`creation process <create-ind>`. Note that the number of rows
   and columns will not mutate beyond the bounds passed to the genetic
   algorithm.

2. Then, with the dimensions of the dataset mutated, each value in the dataset
   is mutated using the same mutation probability. A value is mutated by
   replacing it with a single value sampled from the distribution associated
   with its column.

Parameters for the mutation operator and their definitions can be found
:ref:`here <params-mutation>`.

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
