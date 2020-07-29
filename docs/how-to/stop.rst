Use a stopping condition
------------------------

Stopping conditions allow a EA to terminate before the maximum number of
iterations (generations) have been completed. Using one can save a significant
amount of computational resources, and they can be based on any manner of
things, such as:

- The average fitness of a generation hasn't improved in a number of
  generations.
- The variation between individuals in the population is reasonably low.
- When a best-case solution has been found.

We can include a stopping condition by redefining the
:func:`edo.DataOptimiser.stop` method in a subclass to update the ``converged``
parameter::

    >>> import edo
    >>> import numpy as np
    >>> 
    >>> class MyOptimiser(edo.DataOptimiser):
    ...     def stop(self, tolerance):
    ...         """
    ...         Stop if the population fitness variance is less than
    ...         ``tolerance``.
    ...         """
    ...         fitness_variance = np.var(self.pop_fitness)
    ...         self.converged = fitness_variance < tolerance

To see this in action, consider the example from the `first tutorial`_::

    >>> from edo.distributions import Uniform
    >>> 
    >>> Uniform.param_limits["bounds"] = [-1, 1]
    >>> families = [edo.Family(Uniform)]
    >>> 
    >>> def xsquared(ind):
    ...     return ind.dataframe.iloc[0, 0] ** 2
    >>> 
    >>> opt = MyOptimiser(xsquared, 100, [1, 1], [1, 1], families, max_iter=5)

Now we can run the algorithm as normal, and with an appropriate value of
``tolerance``, it will stop before the maximum number of iterations::

    >>> _ = opt.run(random_state=0, stop_kwargs={"tolerance": 1e-6})
    >>> opt.generation
    4

.. _first tutorial: ../tutorial/xsquared.ipynb
