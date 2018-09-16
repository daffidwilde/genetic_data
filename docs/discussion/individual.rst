Individuals
-----------

.. _representation:

Representation
++++++++++++++

At the beginning of a GA, an population of individuals is generated. Typically,
these individuals are created by randomly sampling parameters from a space at
random -- though `other methods exist
<https://en.wikipedia.org/wiki/Latin_hypercube_sampling>`_. Each individual
represents a solution to the problem at hand and would
classically be given in the form of a bit string to imitate a genetic
chromosome. This traditional approach is what many would consider the defining
feature of a genetic algorithm over some other evolutionary process but we
believe that the inclusion of the biological operators (selection, crossover and
mutation) are the most important features of a genetic algorithm.

Since Edo deals with the creation of datasets, more meaningful manipulation of
individuals can be done when they are left as datasets -- as opposed to
converting the information to obtain a bit string. For computational reasons,
individuals are represented by both a physical dataset with rows and columns,
and a list of probability distributions. Each of these distributions acts a set
of instructions on how to create, inherit from and mutate the values of the
corresponding column in the dataset. These quantities are contained in a single
:code:`namedtuple` object.

.. note::

   If you're unfamiliar with how named tuples work, they are somewhere between a
   class and a tuple that is both lightweight and mutable. If you want to learn
   more, you can go and read the `documentation
   <https://docs.python.org/2/library/
   collections.html#collections.namedtuple>`_.

.. _create-ind:

Creation
++++++++

The parameter space from which individual datasets are generated is defined by
the :code:`row_limits`, :code:`col_limits` and :code:`pdfs` parameters in
:code:`run_algorithm`. The first two describe the dimensional limits of the
dataset (i.e. how tall or wide it can be) while the latter is a pool of families
of probability distributions with which to fill in the dataset.

Once the number of rows and columns have been sampled, a family of distributions
is sampled from the pool, and an instance of that distribution is created --
once for each column. These instances are stored for future operations, and
values are sampled from them to fill their respective column in the dataset. A
schematic of how this is done is given below:

.. image:: ../_static/individual.svg
    :alt: A schematic for the creation of an individual
    :width: 100 %
    :align: center
