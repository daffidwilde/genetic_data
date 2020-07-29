Access information about an individual
--------------------------------------

Individuals are defined by three things in ``edo``: a dataset, metadata about
the distributions used to form the columns of that dataset, and a pseudo-random
number generator for sampling from those distributions.

You can access each of these objects in the same way you would with attributes.
To demonstrate, let's create an individual::

    >>> import numpy as np
    >>> from edo import Family
    >>> from edo.individual import create_individual
    >>> from edo.distributions import Normal, Poisson
    >>> 
    >>> state = np.random.RandomState(0)
    >>> 
    >>> individual = create_individual(
    ...     row_limits=[3, 3],
    ...     col_limits=[4, 4],
    ...     families=[Family(Normal), Family(Poisson)],
    ...     weights=None,
    ...     random_state=state,
    ... )

Then the dataframe can be accessed like this::

    >>> individual.dataframe
               0   1  2          3
    0   2.455133   8  2  13.795999
    1   2.473556  13  0  -2.606494
    2 -10.151318  10  2  -3.112364

And the metadata like this::

    >>> individual.metadata
    [Normal(mean=1.86, std=8.44), Poisson(lam=8.92), Poisson(lam=0.99), Normal(mean=-1.23, std=9.88)]
