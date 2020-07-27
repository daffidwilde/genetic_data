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

The ``edo`` library provides an evolutionary algorithm that optimises any
real-valued function over a subset of the space of all possible datasets that we
call `Evolutionary Dataset Optimisation`. The output of the algorithm is a bank
of effective datasets for which the provided function performs well that can
then be studied.

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

Installation
============

The ``edo`` library requires Python 3.5+ and is ``pip``-installable::

    $ python -m pip install edo

To install from source then clone the GitHub repo::

    $ git clone https://github.com/daffidwilde/edo.git
    $ cd edo
    $ python setup.py install

Publications and documentation
==============================

Full documentation for the library is available at https://edo.readthedocs.io.

An article on the theory behind the algorithm has been published:

    Wilde, H., Knight, V. & Gillard, J. Evolutionary dataset optimisation:
    learning algorithm quality through evolution. *Appl Intell* **50**,
    1172-1191 (2020). https://doi.org/10.1007/s10489-019-01592-4

Citation instructions
=====================

Citing the library
------------------

Please use the following to cite the library::

    @misc{edo-library,
        author = {{The EDO library developers}},
        title = {edo: <RELEASE TITLE>},
        year = <RELEASE YEAR>,
        doi = {<DOI INFORMATION>},
        url = {http://doi.org/<DOI INFORMATION>}
    }

To check the relevant details (i.e. ``RELEASE TITLE``, ``RELEASE YEAR`` and
``DOI NUMBER``) head to the library's Zenodo page:

.. image:: https://zenodo.org/badge/139703799.svg
   :target: https://zenodo.org/badge/latestdoi/139703799

Citing the paper
----------------

If you wish to cite the paper, then use the following::

    @article{edo-paper,
        title = {Evolutionary dataset optimisation: learning algorithm quality
                 through evolution},
        author = {Wilde, Henry and Knight, Vincent and Gillard, Jonathan},
        journal = {Applied Intelligence},
        year = 2020,
        volume = 50,
        pages = {1172--1191},
        doi = {10.1007/s10489-019-01592-4},
    }

Contributing to the library
===========================

Contributions are always welcome whether they come in the form of providing a
fix for a current `issue <https://github.com/daffidwilde/edo/issues>`_,
reporting a bug or implementing an enhancement to the library code itself. Pull
requests (PRs) will be reviewed and collaboration is encouraged.

To make a contribution via a PR, follow these steps:

1. Make a fork of the `GitHub repo <https://github.com/daffidwilde/edo>`_ and
   clone your fork locally::

        $ git clone https://github.com/<your-username>/edo.git

2. Install the library in development mode. If you use Anaconda, there is a
   ``conda`` environment file (``environment.yml``) with all of the development
   dependencies::

        $ cd edo
        $ conda env create -f environment.yml
        $ conda activate edo-dev
        $ python setup.py develop

3. Make your changes and write tests to go with them. Ensure that they pass and
   you have 100% coverage::
   
        $ python -m pytest --cov=edo --cov-fail-under=100 tests

4. Push to your fork and open a pull request.
