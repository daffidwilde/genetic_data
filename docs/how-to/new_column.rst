Implement a new column distribution
-----------------------------------

You are not limited to use only the distributions that are currently
implemented. In fact, because of the way in which individuals are :ref:`operated
<operators>` on in Edo, the inclusion of a new distribution class makes no
difference to how the genetic algorithm will run, and the process of doing so is
relatively painless.

Say, for example, you wanted to implement a triangular distribution class. The
first step would be to import the :code:`Distribution` base class and any
requisite libraries::

   >>> import numpy as np
   >>> from edo.pdfs import Distribution

Now, you define your class as normal, inheriting from the :code:`Distribution`
class. The only requirements and constraints on your class are that:

- there must be class attribute :code:`name` giving the name of the
  distribution;
- there must be two class attributes :code:`hard_limits` and
  :code:`param_limits` giving name and limits of the parameters in the
  distribution;
- it must have a :code:`sample` method that takes itself and :code:`nrows` as
  argument;
- the only attributes defined in its :code:`__init__` are the parameters of that
  particular instance of the distribution, not the limits on its parameters as
  they should be class attributes.

So, bearing that in mind, a triangular distribution class would look something
like this::

   >>> class Triangular(Distribution):
   ...     """ A continuous column distribution given by the triangular
   ...     distribution. """
   ... 
   ...     name = 'Triangular'
   ...     hard_limits = {"bounds": [-10, 10]}
   ...     param_limits = {"bounds": [-10, 10]}
   ... 
   ...     def __init__(self):
   ... 
   ...         limits = self.param_limits["bounds"]
   ...         self.left = np.random.uniform(*limits)
   ...         self.right = np.random.uniform(self.left, limits[1])
   ...         self.mode = np.random.uniform(self.left, self.right)
   ... 
   ...     def sample(self, nrows):
   ...         """ Take a sample of size :code:`nrows` from the triangular
   ...         distribution with the given left, mode and right parameters. """
   ...
   ...         return np.random.triangular(
   ...             self.left, self.mode, self.right, size=nrows
   ...         )

.. warning::
    This base class comes with a :code:`sample` method that raises a
    :code:`NotImplementedError` if you don't write your own method.
