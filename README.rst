.. image:: https://img.shields.io/pypi/v/edo.svg
   :target: https://pypi.org/project/edo/

.. image:: https://github.com/daffidwilde/edo/workflows/CI/badge.svg
   :target: https://github.com/daffidwilde/edo/actions?query=workflow%3ACI+branch%3Amain

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/ambv/black

Evolutionary Dataset Optimisation
*********************************

A library for generating artificial datasets through evolution.
===============================================================

This library contains an evolutionary algorithm that optimises any real-valued
function over a subset of the space of all possible datasets. The output of the
algorithm is a bank of effective datasets for which the provided functio
n performs well that can then be studied.

The applications of this method are varied but an important and relevant one is
in learning an algorithm's strengths and weaknesses.

When determining the quality of an algorithm, the standard route is to run the
comparable algorithms on a finite set of existing (or newly simulated) datasets
and calculating some metric. The algorithm(s) with the smallest value of this
metric are chosen to be the best performing.

An issue with this approach is that it pays little regard to the reliability
and quality of the datasets being used, which begs the question: what makes
a dataset "good" for an algorithm? Or, why is it that an algorithm performs well
on some datasets but not others?

By passing the objective function of the algorithm to the ``edo.DataOptimiser``
class, questions like these can be answered by studying the properties of the
resultant datasets. Beyond that, a combination of objective functions could be
used to determine how an algorithm performs against any number of other
algorithms. A comprehensive description of the evolutionary algorithm and an
examplar case study is available at https://doi.org/10.1007/s10489-019-01592-4.

Publications and documentation
==============================

Full documentation for the library is available at https://edo.readthedocs.io.

An article on the theory behind the algorithm has been published:

|    Wilde, H., Knight, V. & Gillard, J. Evolutionary dataset optimisation:
     learning algorithm quality through evolution. *Appl Intell* **50**,
     1172–1191 (2020). https://doi.org/10.1007/s10489-019-01592-4

.. include:: CITATION.rst

.. include:: INSTALLATION.rst

.. include:: CONTRIBUTING.rst
