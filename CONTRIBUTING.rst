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
