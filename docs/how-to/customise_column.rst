Customise column distributions
------------------------------

As is discussed in the :ref:`tutorials <refs-tutorials>`, the distribution
classes that are passed to the :code:`pdfs` parameter are actually families of
distributions. The limits on the parameters of those distributions can be
altered.

You can take any distribution that is :ref:`currently implemented
<dists>` and set the limits on its parameters before running the genetic
algorithm. Consider the family of normal distributions. The default bounds are
-10 and 10 for the mean, and 0 and 10 for the standard deviation::

    >>> from genetic_data.pdfs import Normal

    >>> Normal.mean_limits, Normal.std_limits
    ([-10, 10], [0, 10])

Changing these bounds is as simple as redefining the class attributes::

    >>> Normal.mean_limits = [-5, 5]
    >>> Normal.std_limits = [0, 1]

    >>> Normal.mean_limits, Normal.std_limits
    ([-5, 5], [0, 10])

Now all instances of normally distributed columns will have mean between -5 and
5, and have standard deviation between 0 and 1.
