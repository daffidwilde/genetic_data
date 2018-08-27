.. _mutate:

Mutation
========

To maintain a level of variety in a population and to force a GA to explore more
of the search space, individuals are mutated immediately after the crossover
process. There are many ways of mutating an individual. The most common is to
take each individual in the population of offspting and run along their alleles,
deciding whether or not to mutate that allele according to a small probability.

In Edo, this probability is controlled by the parameter
:code:`mutation_prob` in :code:`run_algorithm`. However, the method of mutation
is not quite as simple. An individual is mutated in the following way:

1. The number of rows and columns are mutated by adding and/or removing a line
   from each axis with equal probability. Lines are removed at random, and rows
   and columns are added in the same way as the :ref:`crossover <cross>`
   and :ref:`creation <create-ind>` processes respectively. Note that the
   number of rows and columns will not mutate beyond the bounds passed to the GA
   in the :code:`row_limits` and :code:`col_limits` parameters.

2. Then, with the dimension of the dataset mutated, each value in the dataset is
   mutated with the same mutation probability. A value is mutated by sampling a
   single value from the distribution associated with the current column.

Example
-------

Import the mutation operator::

    >>> from edo.operators import mutation

Set the mutation probability. This is deliberately large to make for a
substantial mutation::

    >>> mutation_prob = 0.5

Mutate the offspring that was just created::

    >>> mutant = mutation(
    ...     offspring,
    ...     prob=mutation_prob,
    ...     row_limits=row_limits,
    ...     col_limits=col_limits,
    ...     pdfs=pdfs,
    ...     weights=None
    ... )

This gives the following mutated dataset:

.. csv-table:: The mutant
   :file: mutant.csv
   :align: center

.. include:: mutant.rst
