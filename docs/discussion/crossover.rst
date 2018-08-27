.. _cross:

Crossover
=========

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
-------

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
