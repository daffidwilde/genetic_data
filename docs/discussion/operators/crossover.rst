.. _cross:

Crossover
=========

Crossover operators allow the preservation of preferable characteristics in
genetic algorithms. They take two individuals (parents) and return one or more
"offspring" individuals.

In Edo, the crossover operator returns exactly one individual from a pair of
parents. Since an individual is defined by its dimensions and its values, an
offspring inherits these characteristics in turn from its parents.

Parameters
----------

.. automodule:: edo.operators
   :members: crossover

Example
-------

Import :mod:`numpy` and the relative pieces from :code:`edo`::

    >>> import numpy as np
    >>> from edo.individual import create_individual
    >>> from edo.operators import crossover
    >>> from edo.pdfs import Poisson

Set a seed::

    >>> np.random.seed(0)

Define the constraints and parameters of the simulation::

    >>> row_limits, col_limits = [1, 3], [1, 5]
    >>> pdfs = [Poisson]

Generate two individuals::

    >>> parents = [
    ...     create_individual(row_limits, col_limits, pdfs) for _ in range(2)
    ... ]

These individuals' dataframes look like this:

.. csv-table:: The first parent
   :file: parent_1.csv
   :align: center

.. csv-table:: The second parent
   :file: parent_2.csv
   :align: center

.. include:: parents.rst

Now, we set the cut-off probability and apply the crossover::

    >>> crossover_prob = 0.5
    >>> offspring = crossover(*parents, col_limits, pdfs, crossover_prob)

Then :code:`offspring` has the following dataset:

.. csv-table:: The offspring
   :file: offspring.csv
   :align: center

.. include:: offspring.rst
