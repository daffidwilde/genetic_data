.. _smoothing:

Smoothing
---------

Sometimes objective functions are not definite and have a stochastic element to
them. An example of this is used in the :ref:`second tutorial <tutorial-ii>`.
However, these cases can be handled by the algorithm in EDO by use of a
technique called smoothing.

There are many different ways of implementing smoothing within a fitness
function; in the tutorial several repetitions (samples) are taken and
amalgamated to give a representative value for the objective function. Depending
on the problem domain you are investigating, more robust or specific methods may
be available to you such as `Recursive Bayesian estimation
<https://en.wikipedia.org/wiki/Recursive_Bayesian_estimation>`.

Without smoothing, if your function is stochastic, then you will not obtain a
fair or truthful value for the fitness of an individual dataset.
