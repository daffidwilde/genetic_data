Customise the selection process
-------------------------------

You can alter the selection discipline of the EA using two parameters in
:class:`edo.DataOptimiser`: ``best_prop`` and ``lucky_prop``. These control
how many of the best individuals and any lucky (random) individuals should be
selected respectively.

For example, say we wanted to see the effect of selecting parents purely at
random in each generation. Then we would set ``best_prop`` to be zero, and
``lucky_prop`` to be some value between 0 and 1::

    >>> import edo
    >>> from edo.distributions import Uniform
    >>> 
    >>> def xsquared(ind):
    ...     return ind.dataframe.iloc[0, 0] ** 2
    >>> 
    >>> opt = edo.DataOptimiser(
    ...     xsquared,
    ...     100,
    ...     [1, 1],
    ...     [1, 1],
    ...     [edo.Family(Uniform)],
    ...     best_prop=0,
    ...     lucky_prop=0.25,
    ... )
