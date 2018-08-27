Set a seed
==========

Seeds are controlled by the :code:`seed` parameter in :code:`run_algorithm` and
should be integer.

.. note::
   Without one, the GA here will just fall back on Numpy's innate pseudo-random
   number generator making any results inconsistent between runs.
