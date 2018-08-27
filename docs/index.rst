.. Edo documentation master file, created by
   sphinx-quickstart on Mon Jul 16 12:37:18 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Edo's documentation!
===============================

Edo provides a framework for **E**\volutionary **D**\ataset **O**\ptimisation.
That is, optimising the generation of artificial datasets
through the application of evolutionary dynamics and a genetic algorithm (GA).

Consider a specific algorithm, and its objective function. Edo allows you to
pass that function to a GA as its fitness function. The GA will then go on to
create generations of datasets for which that algorithm performs increasingly
well at.

Through this approach, a user can not only create banks of effective datasets
for their own use, but is also able to determine and study the preferred
characteristics of such datasets.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   tutorial/index.rst
   how-to/index.rst
   discussion/index.rst
   reference/index.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
