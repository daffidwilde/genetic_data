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

In GeneticData, a proportion of the best performing datasets are taken from the
population where the meaning of "best" is controlled by the `maximise`
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
"offspring" individuals. In GeneticData, the crossover operator returns exactly
one individual from a pair of parents.

An individual is defined by its dimensions and its values. So, in the crossover
of two individuals, an offspring inherits the same characteristics from its
parents according to a probability defined by the `crossover_prob` parameter in
:code:`run_algorithm`. This probability indicates the probability with which to
inherit from the first parent rather than the second, and is 0.5 by default.

The process of crossing two datasets is:

0. Determine the widest parent. If they are the same width, this is unimportant.

1. Choose the number of rows (:math:`n`) and number of columns (:math:`k`) from
   the two parents. Each choice is independent of the other.

2. If :math:`k` was inherited from the thinner parent, then inherit columns from
   either parent according to `crossover_prob` as needed. Otherwise, if the new
   individual will be the same width as the wider parent, inherit from either
   parent where possible (up to the last column of the thinner parent) and then
   simply inherit from the wider parent.

3. Now that the individual is of the correct width, its length needs to be
   corrected. This is done by adding and removing rows as needed. Rows to be
   removed are selected at random. Rows are added to the end of a dataset by
   adding a row of :code:`NaN` values. These are then filled in by sampling a
   value from each column. If, for whatever reason, there are no real values in
   a column, then the entire column is replaced by a new column of the correct
   length. This is done in the same way as the
   :ref:`initial creation <create-ind>` of an individual.

Example
+++++++

Import the relative pieces from :code:`genetic_data`::

    >>> import numpy as np
    >>> from genetic_data.components import create_individual
    >>> from genetic_data.operators import crossover
    >>> from genetic_data.pdfs import Normal

Set a seed::

    >>> np.random.seed(1)

Define the constraints and initial parameters of the simulation::

    >>> row_limits, col_limits = [1, 3], [1, 5]
    >>> pdfs = [Normal]

Generate two individuals::

    >>> parents = [create_individual(row_limits, col_limits, pdfs) \
    ...            for _ in range(2)]

These individuals look like this (every dataset's entries in the following
examples has been rounded to 4 d.p.):

.. csv-table:: The first parent
   :file: parent_1.csv
   :align: center

.. csv-table:: The second parent
   :file: parent_2.csv
   :align: center

Finally, apply the crossover::

    >>> offspring = crossover(*parents, prob=0.5)

Then :code:`offspring` is the following dataset:

.. csv-table:: The offspring
   :file: offspring.csv
   :align: center

.. _mutate:

Mutation
--------

To maintain a level of variety in a population and to force a GA to explore more
of the search space, individuals are mutated immediately after the crossover
process. There are many ways of mutating an individual. The most common is to
take each individual in the population of offspting and run along their alleles,
deciding whether or not to mutate that allele according to a small probability.

In GeneticData, this probability is controlled by the parameter
:code:`mutation_prob` in :code:`run_algorithm`. However, the method of mutation
is not quite as simple. An individual is mutated in the following way:

1. The number of rows and columns are mutated by adding or removing a line from
   each axis with equal probability either way. The process of adding and
   removing lines is the same as in the :ref:`crossover <cross>` process. Note
   that the number of rows and columns will not mutate beyond the bounds passed
   to the GA in the :code:`row_limits` and :code:`col_limits` parameters.

2. Then, with the dimension of the dataset mutated, each value in the dataset is
   mutated with the same mutation probability. A value is mutated by sampling a
   single value from the normal distribution centred at the current value and
   with standard deviation given by the parameter :code:`sigma`. This stops the
   mutation process from changing an individual too drastically by using smaller
   values of :code:`sigma`. Though, more dramatic mutation can be encouraged by
   setting this parameter (and `mutation_prob`) to be higher.

Example
+++++++

Import the mutation operator::

    >>> from genetic_data.operators import mutation

Set the mutation parameters. These are deliberately large to guarantee a
substantial mutation::

    >>> mutation_prob = 1.
    >>> sigma = 10.

Mutate the offspring that was just created::

    >>> mutant = mutation(
    ...     offspring,
    ...     prob=mutation_prob,
    ...     row_limits=row_limits,
    ...     col_limits=col_limits,
    ...     pdfs=pdfs,
    ...     weights=None,
    ...     sigma=10.
    ... )

This gives the following mutated dataset:

.. csv-table:: The mutant
   :file: mutant.csv
   :align: center
