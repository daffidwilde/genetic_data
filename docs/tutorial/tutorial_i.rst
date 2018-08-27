.. _refs-tutorial-i:

Optimising a function
=====================

Suppose we want to optimise the function :math:`f(x) = x^2` across the whole
real line. We can consider each :math:`x` to be a dataset with exactly one row
and one column like so:

.. table::
   :align: center

   +-----------+-----------+
   |           | column 0  |
   +===========+===========+
   | **row 0** | :math:`x` |
   +-----------+-----------+

For the sake of this example, let us assume our initial population has 100
individuals in it, each of which is normally distributed.

Let us allow these normal distributions to randomly sample their mean and
standard deviation within the bounds [-10, 10] and [0, 10] respectively.

The hope is that the GA will converge to something near the optimum of
:math:`f(x)` relatively quickly, so we will only allow the algorithm to run for
a maximum of 5 iterations.

Formulation
-----------

So, to formulate and solve this problem in Edo we do the following:

First, import the library::

    >>> import edo

Define our fitness function::

    >>> def x_squared(df):
    ...     return df.iloc[0, 0] ** 2

Set up and run the GA::

    >>> pop, fit, all_pops, all_fits = edo.run_algorithm(
    ...     fitness=x_squared,
    ...     size=100,`
    ...     row_limits=[1, 1],
    ...     col_limits=[1, 1],
    ...     max_iter=5,
    ...     maximise=False,
    ...     seed=0
    ... )

The outputs of :code:`run_algorithm` are the final population and its associated
fitness vector, as well as lists of all populations and fitness scores in the
simulation. These are held in the variables :code:`pop`, :code:`fit`,
:code:`all_pops` and :code:`all_fits` respectively.

.. note::
    Here, many of the parameters for the GA are using their default settings.
    Discussion on how to customise these parameters is found in the
    :ref:`how-to` section, and their definitions are listed in :ref:`params`.

Visualising results
-------------------

To see the results of the GA (and whether it worked as expected) we will plot
the fitnesses of all the individuals at each epoch (timestep).

We'll do this using Matplotlib and NumPy, so let's import them first::

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

Then using Matplotlib we can superimpose our theoretical fitness function (the
solid blue line) with all of our observed fitness scores (the orange scatter
points)::

    >>> fig, (top, middle, bottom) = plt.subplots(
    ...     nrows=3,
    ...     ncols=2,
    ...     figsize=(30, 45),
    ...     dpi=300,
    ...     sharex=True,
    ...     sharey=True
    ... )

    >>> xs = np.linspace(-25, 25, 101)

    >>> for i in range(6):
    ...
    ...     if i < 2:
    ...         axes = top
    ...     elif i < 4:
    ...         axes = middle
    ...     else:
    ...         axes = bottom
    ...
    ...     j = i % 2
    ...     data = [[ind.dataframe.iloc[0, 0] for ind in all_pops[i]], all_fits[i]]
    ...
    ...     axes[j].plot(xs, xs ** 2, lw=3, zorder=-1)
    ...     axes[j].scatter(*data, s=200, color='orange')
    ...
    ...     axes[j].set_title(f'Fitness scores in epoch {i}', size=24, pad=25)
    ...     if i in [4, 5]:
    ...         axes[j].set_xlabel(r'$x$', size=24)
    ...     if i in [0, 2, 4]:
    ...         axes[j].set_ylabel('Fitness', size=24)

    >>> plt.tight_layout(pad=5)
    >>> plt.show()

The above code should give a figure like this:

.. image:: ../_static/tutorial_i_plot.png
   :width: 100 %
   :align: center
   :alt: Fitness scores of every individual

So the GA has successfully started converging towards zero. Good news!
