Customise the mutation process
------------------------------

The mutation process can be altered by adjusting the mutation probability. That
is done using the :code:`mutation_prob` parameter in :code:`run_algorithm`.

Using the example from the :ref:`first tutorial <refs-tutorial-i>`, we can (for
instance) remove all mutation by setting this parameter to be zero::

    >>> import edo
    >>> from edo.pdfs import Normal

    >>> def x_squared(df):
    ...     return df.iloc[0, 0] ** 2

    >>> pop, fit, all_pops, all_fits = edo.run_algorithm(
    ...     fitness=x_squared,
    ...     size=100,
    ...     row_limits=[1, 1],
    ...     col_limits=[1, 1],
    ...     pdfs=[Normal],
    ...     max_iter=100,
    ...     mutation_prob=0,
    ...     seed=0
    ... )
