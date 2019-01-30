Customise the mutation process
------------------------------

The mutation process can be altered in three ways:

1. Setting the initial mutation probability
2. Adjusting (dwindling) the mutation probability over time
3. Compacting the mutation space around the best individuals

Below are some quick examples of how to do these things.

Setting the initial probability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is done using the :code:`mutation_prob` parameter in :code:`run_algorithm`.

Using the example from the :ref:`first tutorial <refs-tutorial-i>`, we can (for
instance) remove all mutation by setting this parameter to be zero::

    >>> import edo

    >>> def x_squared(df):
    ...     return df.iloc[0, 0] ** 2

    >>> pop, fit, all_pops, all_fits = edo.run_algorithm(
    ...     fitness=x_squared,
    ...     ...
    ...     mutation_prob=0
    ... )

Dwindling mutation probability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes a genetic algorithm can be thrown off once it has started converging.
The purpose of the mutation process is to do this deliberately. However, as the
GA progresses, mutation can make the population unpredictable or noisy.

To combat this, a function for dwindling (or incrementing, if that's your thing)
the mutation probability can be passed to the :code:`dwindle` parameter.

This function must take only the current mutation probability and the current
iteration as argument, and must return the new mutation probability::

    >>> def half_mutation_prob(mutation_prob, iteration):
    ...     """ Cut the mutation prob in half every 50 iterations. """
    ... 
    ...     if iteration % 50 == 0:
    ...         mutation_prob /= 2
    ...     return mutation_prob

    >>> pop, fit, all_pops, all_fits = edo.run_algorithm(
    ...     fitness=x_squared,
    ...     ...
    ...     dwindle=half_mutation_prob
    ... )

Compacting the mutation space
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The final way to alter the mutation process is to progressively reduce the
mutation space. This is done by reducing the intervals from which distribution
parameters are sampled. The reduced interval is found using the current
parameter values for all the parent columns found in the :ref:`selection process
<selection>`. This method is derived from that set out in [Amirjanov2017]_ and
is controlled using the :code:`compact`::

    >>> pop, fit, all_pops, all_fits = edo.run_algorithm(
    ...     fitness=x_squared,
    ...     ...
    ...     compact=0.75
    ... )
