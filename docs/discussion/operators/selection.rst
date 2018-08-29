Selection
=========

The selection operator is the process by which individuals are chosen from the
current population to act as the "parents" of the next generation. Almost
always, selection operators determine whether an individual should become a
parent based on their fitness.

In Edo, a proportion of the best performing datasets are taken from a
population into the next. You can also choose to include some randomly
selected, or "lucky", individuals to be carried forward with the fittest members
of the population, if there are any still available.

This selection method is a variant of the classic truncation selection method
with a fixed selection proportion. When lucky individuals are included, a level
of noise is introduced which can increase convergence rates [Jebari2013]_.

.. note::
   Taking lucky individuals should be done sparingly as the associated noise
   can throw the genetic algorithm off. The use of this functionality is only
   encouraged for particularly complex contexts where you can't obtain
   satisfactory results otherwise.

Parameters
----------

.. automodule:: edo.operators
   :members: selection
