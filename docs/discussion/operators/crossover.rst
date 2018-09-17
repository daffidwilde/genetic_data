.. _cross:

Crossover
=========

A crossover operator defines how two individuals should be combined to create a
new individual (or individuals). Importantly, the crossover operator allows for
the preservation of preferable characteristics found by the genetic algorithm.

In Edo, the crossover operator returns exactly one individual from a pair of
parents. An individual is defined first, structurally, by its dimensions
and then, qualitively, by its values, and so a new offspring inherits these
characteristics in this order from its parents:

1. Inherit a number of rows and a number of columns from either parent,
   independently, and according to a cut-off probability. This gives an empty
   dataset effectively.

2. Pool together the columns (and respective column information) from both
   parents and sample from them uniformly (and without replacement) to fill in 
   the columns of the offspring's dataset. Whilst doing this, keep track of the
   column information. Now the dataset has values and instructions on how to
   manipulate it.

3. Remove surplus rows as required, and fill in any missing values using the
   corresponding column information. This is now a complete individual.

Parameters for the crossover operator and their definitions can be found
:ref:`here <params-crossover>`.

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
