Access information about an individual
--------------------------------------

Individuals are defined by pairs in Edo. Each pair contains a
:class:`pandas.DataFrame` and a list of metadata about how the columns of that
dataframe were made.

You can access each of these objects in the same way you would with attributes.
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

.. include:: access_dataframe.rst

.. include:: access_metadata.rst
