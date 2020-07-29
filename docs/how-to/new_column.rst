Implement a new column distribution
-----------------------------------

You are not limited to use only the distributions that are currently
implemented in ``edo``. 

Say, for example, you wanted to implement a triangular distribution class. The
first step would be to import the :class:`edo.distributions.Distribution` base
class::

   >>> from edo.distributions import Distribution

Now, you define your class as normal, inheriting from the base class. The
requirements on your class are as follows:

- There must be a class attribute ``name`` giving the name of the
  distribution.
- There must be a class attribute ``dtype`` detailing the preferred data type of
  the distribution.
- There must be a class attribute ``hard_limits`` that gives extreme limits
  on the parameters of the distribution.
- There must be a class attribute ``param_limits`` that gives the original
  limits on the parameters of the distribution.
- It must have a ``sample`` method that takes as argument: itself, an integer
  number of rows ``nrows`` and an instance of ``numpy.random.RandomState``.
- The ``__init__`` takes only an instance of ``numpy.random.RandomState``.
- The only attributes defined in the ``__init__`` are the parameters of that
  particular instance of the distribution and match the keys of
  ``param_limits``.

So, bearing that in mind, a triangular distribution class would look something
like this::

   >>> class Triangular(Distribution):
   ...     """ A continuous column distribution given by the triangular
   ...     distribution. """
   ... 
   ...     name = "Triangular"
   ...     dtype = float
   ...     hard_limits = {"bounds": [-10, 10]}
   ...     param_limits = {"bounds": [-10, 10]}
   ... 
   ...     def __init__(self, random_state):
   ...
   ...         left, mode, right = sorted(
   ...             random_state.uniform(*self.param_limits["bounds"], size=3)
   ...         )
   ...         self.bounds = [left, mode, right]
   ... 
   ...     def sample(self, nrows, random_state):
   ...         """ Take a sample of size ``nrows`` from the triangular
   ...         distribution with the given bounds. """
   ...
   ...         return random_state.triangular(*self.bounds, size=nrows)
