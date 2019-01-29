Use a stopping condition
------------------------

Stopping conditions allow a GA to terminate before the maximum number of
iterations have been completed. Using one can save significant computational
time, and they can be based on all sorts of things, such as:

- if the average fitness is below some threshold;
- if a population of similarly fit individuals has been reached;
- when a best-case solution has been found.

Because there are so many possible approaches you can take, you can pass any
function to the :code:`stop` parameter in :code:`run_algorithm`. The only
constraints on this kind of function are that it takes only the current
population's fitness as argument and returns a boolean value.

Consider the example from the :ref:`first tutorial <refs-tutorial-i>` where we
are trying to optimise the function :math:`f(x) = x^2`. Let us define a stopping
condition on the mean fitness of a population::

    >>> def x_squared(df):
    ...     return df.iloc[0, 0] ** 2

    >>> def mean_stopping(population_fitness, tolerance=0.1):
    ...     """ Return `True` when the mean fitness score is below a tolerance
    ...     level. Otherwise return `False`. The tolerance is 0.1 by default.
    ...     """
    ... 
    ...     mean_score = sum(population_fitness) / len(population_fitness)
    ...     return mean_score < tolerance

Then we simply pass that function to :code:`run_algorithm`, like so::

    >>> import edo
    >>> from edo.pdfs import Normal

    >>> pop, fit, all_pops, all_fits = edo.run_algorithm(
    ...     fitness=x_squared,
    ...     size=100,
    ...     row_limits=[1, 1],
    ...     col_limits=[1, 1],
    ...     pdfs=[Normal],
    ...     stop=mean_stopping,
    ...     max_iter=100,
    ...     seed=0
    ... )
