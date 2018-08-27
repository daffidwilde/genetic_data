Selection
---------

The selection operator is the process by which individuals are chosen from the
current population to act as the "parents" of the next generation. Almost
always, selection operators determine whether an individual should become a
parent based on their fitness.

In Edo, a proportion of the best performing datasets are taken from the
population, where the meaning of "best" is controlled by the :code:`maximise`
parameter. We also allow for the subsequent, random selection of some "lucky"
individuals to be carried forward with the fittest members of the population, if
there are any still available.

These proportions are controlled using the :code:`best_prop` and
:code:`lucky_prop` parameters respectively. 

.. note::
   Taking lucky individuals should be done sparingly as they are chosen at
   random and can throw the genetic algorithm off. The use of this functionality
   is only encouraged for particularly complex contexts where you can't obtain
   good enough results otherwise.
