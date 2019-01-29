Customise the selection process
-------------------------------

You can change the selection discipline of the GA using two parameters in
:code:`run_algorithm` -- :code:`best_prop` and :code:`lucky_prop`. These control
how many of the best individuals and any lucky (random) individuals should be
selected respectively.

For example, say we wanted to see the effect of selecting parents purely at
random in each generation. Then, using the setting of the :ref:`first tutorial
<refs-tutorial-i>`, we can set :code:`best_prop` to be zero, and
:code:`lucky_prop` to be some non-negative value. Let's say that value is 0.25::

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
    ...     best_prop=0,
    ...     lucky_prop=0.25,
    ...     seed=0
    ... )
