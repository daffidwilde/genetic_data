Set a seed
==========

Seeds are controlled by the ``random_state`` parameter in
:func:`edo.DataOptimiser.run` and can be integer or an instance of
``numpy.random.RandomState``.

.. note::
   Without one, the EA here will just fall back on NumPy's innate pseudo-random
   number generator making any results inconsistent between runs.

Taking the example from the `first tutorial`_, we can get different results by
using a different seed::

   >>> import edo
   >>> from edo.distributions import Uniform
   >>> 
   >>> Uniform.param_limits["bounds"] = [-1, 1]
   >>> families = [edo.Family(Uniform)]
   >>> 
   >>> def xsquared(ind):
   ...     return ind.dataframe.iloc[0, 0] ** 2
   >>> 
   >>> opt = edo.DataOptimiser(
   ...     fitness=xsquared,
   ...     size=100,
   ...     row_limits=[1, 1],
   ...     col_limits=[1, 1],
   ...     families=families,
   ...      max_iter=5,
   ... )
   >>> _, fit_history = opt.run(random_state=0)
   >>> fit_history.head()
       fitness  generation  individual
   0  0.133711           0           0
   1  0.058883           0           1
   2  0.682047           0           2
   3  0.315748           0           3
   4  0.011564           0           4
   >>> 
   >>> opt = edo.DataOptimiser(
   ...     fitness=xsquared,
   ...     size=100,
   ...     row_limits=[1, 1],
   ...     col_limits=[1, 1],
   ...     families=families,
   ...      max_iter=5,
   ... )
   >>> _, fit_history = opt.run(random_state=1)
   >>> fit_history.head()
       fitness  generation  individual
   0  0.095955           0           0
   1  0.154863           0           1
   2  0.096262           0           2
   3  0.081103           0           3
   4  0.011293           0           4

.. _first tutorial: ../tutorial/xsquared.ipynb
