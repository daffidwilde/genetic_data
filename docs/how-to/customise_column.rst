Customise column distributions
------------------------------

As is discussed in :ref:`representation`, the distribution classes that are
passed to the algorithm are actually families of distributions.

In addition to using different distributions, the limits on the parameters of
existing distributions can be altered before passing them to the algorithm.

You can check out `dists` for a list of all the currently implemented
distributions and their default parameters. But for now, let's consider the
family of normal distributions::

    >>> from edo.pdfs import Normal

The default bounds are -10 and 10 for the mean, and 0 and 10 for the standard
deviation::

    >>> Normal.mean_limits, Normal.std_limits
    ([-10, 10], [0, 10])

Changing these bounds is as simple as redefining the class attributes::

    >>> Normal.mean_limits = [-5, 5]
    >>> Normal.std_limits = [0, 1]

    >>> Normal.mean_limits, Normal.std_limits
    ([-5, 5], [0, 1])

Now all instances of normally distributed columns will have mean between -5 and
5, and have standard deviation between 0 and 1.
