Customise the selection process
-------------------------------

It may be be benefitial in your problem to control how parents are selected in
each iteration. You can do that in two ways here by altering the following
parameters in :code:`run_algorithm`:

- :code:`best_prop` -- this is the proportion of the population to skim off
  the top and take forward as parents.
- :code:`lucky_prop` -- after the best individuals are chosen, you can take a
  proportion of "lucky" individuals into the next generation. By default, this
  doesn't happen.

.. note::
    Taking lucky individuals should be done sparingly as they are chosen
    randomly and can throw the GA off. The use of this functionality is only
    encouraged for particularly complex contexts where you can't obtain good
    enough results otherwise.

For example, say we wanted to see the effect of selecting parents at random in
each generation. Then we can set :code:`best_prop` to be zero, and
:code:`lucky_prop` to be some non-negative value. Let's say that value is 0.25::

    >>> pop, fit, all_pops, all_fits = gd.run_algorithm(
    ...     fitness=x_squared,
    ...     size=100,
    ...     row_limits=[1, 1],
    ...     col_limits=[1, 1],
    ...     max_iter=100,
    ...     best_prop=0,
    ...     lucky_prop=0.25,
    ...     seed=0
    ... )
