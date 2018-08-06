Implement a new column distribution
-----------------------------------

Separate from anything to do with this library, if you're interested in an
algorithm whose objective function requires you to count the number of times the
word :code:`cow` appears in a column made up of farm animals, then you
have the freedom to do that.

One of the key benefits of how individuals are represented and operated on in
GeneticData is that the GA will behave in exactly the same way regardless of
what's actually in the dataframe its acting on. That means you can use any
distribution(s) to generate individuals -- including one that samples the names
of farm animals from a list according to some probability distribution.

Implementing such a distribution is easily done and should be done by inheriting
from the :code:`Distribution` base class::

    >>> import numpy as np
    >>> import itertools
    >>> from genetic_data.pdfs import Distribution

    >>> class FarmAnimal(Distribution):
    ...     """ A distribution for sampling the names of some farm animals. """
    ... 
    ...     names = ['cow', 'sheep', 'chicken', 'goose', 'pig']
    ... 
    ...     prob = np.linspace(0, 1, 11)
    ...     probs = [
    ...         np.round(dist, 1)
    ...         for dist in itertools.product(prob, repeat=len(names))
    ...         if sum(dist) == 1
    ...     ]
    ... 
    ...     def __init__(self):
    ... 
    ...         self.probability_dist = np.random.choice(self.probs)
    ... 
    ...     def sample(self, nrows):
    ...         """ Take a sample of size `nrows` from `names` according to the
    ...         probability distribution given by `probability_dist`. """
    ... 
    ...         return np.random.choice(
    ...             self.names, weights=self.probability_dist, size=nrows
    ...         )

.. warning::
    This base class comes with a :code:`sample` method that raises a
    :code:`NotImplementedError` if you don't write your own method.
