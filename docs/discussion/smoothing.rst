.. _smoothing:

Smoothing
---------

Sometimes the fitness function you wish to use will have a stochastic element.
This means that when the fitness of an individual is calculated, it will not
necessarily be the same on another run of the algorithm, or representative at
all. However, this effect can be handled by use of a technique called smoothing.
An example of this is used in the `k-means tutorial`_.

There are many different ways of implementing smoothing within a fitness
function; in the tutorial several repetitions are done using their own random
seeds. These repetitions are then amalgamated to give a representative value for
the objective function. Depending on the problem domain you are investigating,
more robust or specific methods may be available to you such as
`recursive Bayesian estimation
<https://en.wikipedia.org/wiki/Recursive_Bayesian_estimation>`_.

.. _k-means tutorial: ../tutorial/kmeans.ipynb
