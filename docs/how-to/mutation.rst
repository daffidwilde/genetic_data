Customise the mutation process
------------------------------

The mutation process can be altered in three ways:

1. Setting the initial mutation probability
2. Adjusting (dwindling) the mutation probability over time
3. Compacting the mutation space around the best individuals

Below are some quick examples of how to do these things.

Setting the initial probability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is done using the ``mutation_prob`` parameter in
:class:`edo.DataOptimiser`. For instance, we can remove all mutation by setting
this parameter to be zero::

    >>> import edo
    >>> from edo.distributions import Uniform
    >>> 
    >>> def xsquared(ind):
    ...     return ind.dataframe.iloc[0, 0] ** 2
    >>> 
    >>> opt = edo.DataOptimiser(
    ...     xsquared, 100, [1, 1], [1, 1], [edo.Family(Uniform)], mutation_prob=0
    ... )

Dwindling mutation probability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes an evolutionary algorithm can be thrown off once it has started
converging. The purpose of the mutation process is to do this deliberately.
However, as the EA progresses, mutation can make this disruption unhelpful and
the population may become unpredictable or noisy.

To combat this, the :func:`edo.DataOptimiser.dwindle` method can be redefined in
a subclass::

    >>> class MyOptimiser(edo.DataOptimiser):
    ...     def dwindle(self, N=50):
    ...         """ Cut the mutation probability every ``N`` generations. """
    ...         if self.generation % N == 0:
    ...             self.mutation_prob /= 2

Any further arguments for this method should be passed in the ``dwindle``
parameter of :func:`edo.DataOptimiser.run`::

    >>> opt = MyOptimiser(
    ...     xsquared,
    ...     100,
    ...     [1, 1],
    ...     [1, 1],
    ...     [edo.Family(Uniform)],
    ...     max_iter=1,
    ...     mutation_prob=1,
    ... )
    >>> 
    >>> pop_history, fit_history = opt.run(dwindle_kwargs={"N": 1})
    >>> opt.mutation_prob
    0.5

.. _compact:

Compacting the mutation space
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The final way to alter the mutation process is to progressively reduce the
mutation space via :ref:`shrinking <shrinkage>`. This is done using the
``shrinkage`` parameter of :class:`edo.DataOptimiser`::

    >>> opt = edo.DataOptimiser(
    ...     xsquared, 100, [1, 1], [1, 1], [edo.Family(Uniform)], shrinkage=0.9
    ... )
