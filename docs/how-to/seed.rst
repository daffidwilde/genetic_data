Set a seed
==========

Seeds are controlled by the :code:`seed` parameter in :code:`run_algorithm` and
should be integer.

.. note::
   Without one, the GA here will just fall back on Numpy's innate pseudo-random
   number generator making any results inconsistent between runs.

Consider the setting from the :ref:`first tutorial <refs-tutorial-i>`. We will
obtain inconsistent results by running the algorithm twice using two values of
:code:`seed`.

Run the algorithm with one seed::

   >>> import edo
   >>> from edo.pdfs import Normal

   >>> def x_squared(df):
   ...     return df.iloc[0, 0] ** 2

   >>> pop, fit, all_pops, all_fits = edo.run_algorithm(
   ...     x_squared,
   ...     size=100,
   ...     row_limits=[1, 1],
   ...     col_limits=[1, 1],
   ...     pdfs=[Normal],
   ...     max_iter=5,
   ...     seed=0
   ... )

And again, with another seed::

   >>> new_pop, new_fit, new_all_pops, new_all_fits = edo.run_algorithm(
   ...     x_squared,
   ...     size=100,
   ...     row_limits=[1, 1],
   ...     col_limits=[1, 1],
   ...     pdfs=[Normal],
   ...     max_iter=5,
   ...     seed=1
   ... )

.. include:: seed_inds.rst

For further verification, we can look at the fitness progression::

   >>> import numpy as np
   >>> import matplotlib.pyplot as plt
   >>> from matplotlib.patches import Patch

   >>> fig, ax = plt.subplots(1, figsize=(32, 12), dpi=300)

   >>> width = 0.3
   >>> epsilon = 0.01
   >>> positions = np.arange(len(all_fits))
   >>> shift = 0.5 * width + epsilon

   >>> old = ax.boxplot(
   ...     all_fits,
   ...     positions=positions - shift,
   ...     widths=width,
   ...     patch_artist=True
   ... )

   >>> new = ax.boxplot(
   ...     new_all_fits,
   ...     positions=positions + shift,
   ...     widths=width,
   ...     patch_artist=True
   ... )

   >>> xticks = ax.set_xticks(positions)
   >>> xticklabels = ax.set_xticklabels(positions)
   >>> ax.set_yscale('log')
   >>> xlabel = ax.set_xlabel('Epoch', fontsize=24)
   >>> ylabel = ax.set_ylabel(r'$\log (f(x))$', fontsize=24)
   ...

   >>> for plot, colour in zip([old, new], ['lightblue', 'lightpink']):
   ...     for patch in plot['boxes']:
   ...         patch.set_facecolor(colour)

   >>> for label in xticklabels + ax.get_yticklabels():
   ...     label.set_fontsize(20)

   >>> legend = ax.legend(
   ...     handles=[Patch(color='lightblue'), Patch(color='lightpink')],
   ...     labels=['Seed 0', 'Seed 1'],
   ...     fontsize=24
   ... )

   >>> plt.show()

.. image:: ../_static/seed.svg
   :width: 100 %
   :align: center
   :alt: Results for two different seeds
