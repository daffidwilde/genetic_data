.. _ref-operators:

Operators
=========

Most genetic algorithms make use of three operators to move from the current
population to the next. These are *selection*, *crossover* and *mutation*.
Descriptions of the operators in this GA are given below.

Selection
---------

The selection operator is the process by which individuals are chosen from the
current population to act as the "parents" of the next generation. Almost
always, selection operators determine whether an individual should become a
parent based on their fitness.

In Edo, a proportion of the best performing datasets are taken from the
population where the meaning of "best" is controlled by the :code:`maximise`
parameter. We also allow for the random selection of some "lucky" individuals to
be carried forward with the fittest members of the population, if there are any
still available.

These proportions are controlled using the :code:`best_prop` and
:code:`lucky_prop` parameters respectively.

.. note::
    The best individuals are always chosen before lucky ones. This does mean
    that random selection can still be forced by setting :code:`best_prop` to be
    0.

.. _cross:

Crossover
---------

Crossover operators take two individuals (parents) and return one or more
"offspring" individuals. In Edo, the crossover operator returns exactly
one individual from a pair of parents.

An individual is defined by its dimensions and its values. So, in the crossover
of two individuals, an offspring inherits the same characteristics from its
parents according to a probability defined by the :code:`crossover_prob`
parameter in :code:`run_algorithm`. This probability indicates the probability
with which to inherit from the first parent rather than the second, and is 0.5
by default.

The process of crossing two datasets is:

0. Determine the widest parent. If they are the same width, this is unimportant.

1. Choose the number of rows (:math:`n`) and number of columns (:math:`k`) from
   the two parents. Each choice is independent of the other.

2. If :math:`k` was inherited from the thinner parent, then inherit columns from
   either parent according to :code:`crossover_prob` as needed. Otherwise, if
   the new individual will be the same width as the wider parent, inherit from
   either parent where possible (up to the last column of the thinner parent)
   and then simply inherit from the wider parent. Note that when a column is
   inherited, the associated probability distribution is also inherited.

3. Now that the individual is of the correct width, its length needs to be
   corrected. This is done by adding and removing rows as needed. Rows to be
   removed are selected at random, and rows are added by appending an empty row
   to the end of a dataset. This row is then filled in by sampling a
   value from the distribution associated with each column.

Example
+++++++

Import the relative pieces from :code:`genetic_data`::

    >>> import numpy as np
    >>> from edo.individual import create_individual
    >>> from edo.operators import crossover
    >>> from edo.pdfs import Poisson

Set a seed::

    >>> np.random.seed(0)

Define the constraints and initial parameters of the simulation::

    >>> row_limits, col_limits = [1, 3], [1, 5]
    >>> pdfs = [Poisson]

Generate two individuals::

    >>> parents = [create_individual(row_limits, col_limits, pdfs) \
    ...            for _ in range(2)]

These individuals' dataframes look like this:

.. csv-table:: The first parent
   :file: parent_1.csv
   :align: center

.. csv-table:: The second parent
   :file: parent_2.csv
   :align: center

.. include:: parents.rst

Now, we apply the crossover::

    >>> offspring = crossover(*parents, prob=0.5)

Then :code:`offspring` is the following dataset:

.. csv-table:: The offspring
   :file: offspring.csv
   :align: center

.. include:: offspring.rst

.. _mutate:

Mutation
--------

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
+++++++

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
