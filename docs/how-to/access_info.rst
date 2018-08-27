Access information about an individual
--------------------------------------

Individuals are defined by pairs in this GA. Each pair contains a
:code:`pandas.DataFrame` and a list of metadata about how the columns of that
dataframe were made.

You can access each of these things in the same way you would class attributes.
To demonstrate, let's create an individual::

    >>> import numpy as np
    >>> from edo.individual import create_individual
    >>> from edo.pdfs import Normal, Poisson

    >>> np.random.seed(0)
    >>> individual = create_individual(
    ...     row_limits=[3, 3],
    ...     col_limits=[4, 4],
    ...     pdfs=[Normal, Poisson],
    ...     weights=None
    ... )

Then the dataframe can be accessed like this::

    >>> individual.dataframe

.. csv-table::
   :file: access_dataframe.csv
   :align: center

And the metadata like this::

    >>> individual.metadata
    [Normal(mean=2.06, std=5.45), Poisson(lambda=9.64),
     Poisson(lambda=6.67), Normal(mean=3.13, std=1.38)]
