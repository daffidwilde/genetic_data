Access information about an individual
--------------------------------------

Individuals are defined by pairs in this GA. Each pair is contained in a
:code:`namedtuple` with two fields:

- a dataframe
- some column metadata related to how that dataframe is generated

If you're unfamiliar with how named tuples work, they're similar to classes but
more lightweight, whilst like a standard Python :code:`tuple` that is mutable.
To learn more, you can go and read the `documentation
<https://docs.python.org/2/library/collections.html#collections.namedtuple>`_.

You can access each of the fields in the same way you would class attributes. To
demonstrate, let's create an individual::

    >>> import numpy as np
    >>> from genetic_data.creation import create_individual
    >>> from genetic_data.pdfs import Normal, Poisson

    >>> np.random.seed(0)
    >>> individual = create_individual(
    ...     row_limits=[3, 3],
    ...     col_limits=[4, 4],
    ...     pdfs=[Normal, Poisson]
    ... )

Then the dataframe can be accessed like this::

    >>> individual.dataframe

.. csv-table::
   :file: access_dataframe.csv
   :align: center

And the metadata like this::

    >>> individual.column_metadata
    [Normal(mean=2.06, std=5.45), Poisson(lambda=9.64),
     Poisson(lambda=6.67), Normal(mean=3.13, std=1.38)]
