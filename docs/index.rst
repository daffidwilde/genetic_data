.. Edo documentation master file, created by
   sphinx-quickstart on Mon Jul 16 12:37:18 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Edo's documentation!
===============================

Edo provides a framework for optimising the generation of artificial datasets
through the application of a evolutionary dynamics and genetic algorithms (GAs).

Consider a specific algorithm, and its objective function. This GA can take that
objective function as its own fitness function to create generations of datasets
for which that algorithm performs increasingly well at.

Through this approach, a user can not only create banks of effective datasets
for their own use, but can also be able to determine and study the preferred
characteristics of such datasets. This is made easy for a user by considering
each individual in a population to be a pair: one part being the dataset itself,
and the other being a list of distributions from which the column's values are
generated.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   tutorial/index.rst
   how-to/index.rst
   reference/index.rst
   discussion/index.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
