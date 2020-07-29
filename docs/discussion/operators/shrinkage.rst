.. _shrinkage:

Shrinkage
---------

It is possible to reduce the search space of the algorithm forcibly by including
some *shrinkage* or *compacting*. Under this operation, each distribution family
has its parameter limits reduced by those present in the parents from a
generation according to a power law presented in [AS17]_.

.. warning::

    This can produce reductive results and is not recommended in normal use.
