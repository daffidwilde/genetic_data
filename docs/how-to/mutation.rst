Customise the mutation process
------------------------------

Since the way in which individuals are :ref:`mutated <mutate>` is fixed, the
only way you can alter the mutation process is using the mutation probability.
That is done using the :code:`mutation_prob` parameter in :code:`run_algorithm`.

Using the example above, we can (for instance) remove all mutation by setting
this parameter to be zero::

    >>> import edo
    >>> pop, fit, all_pops, all_fits = edo.run_algorithm(
    ...     fitness=x_squared,
    ...     size=100,
    ...     row_limits=[1, 1],
    ...     col_limits=[1, 1],
    ...     max_iter=100,
    ...     mutation_prob=0,
    ...     seed=0
    ... )
