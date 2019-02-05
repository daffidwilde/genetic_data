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

    >>> Normal.param_limits
    {"mean": [-10, 10], "std": [0, 10]}

Changing these bounds is as simple as redefining the class attributes::

    >>> Normal.param_limits["mean"] = [-5, 5]
    >>> Normal.param_limits["std"] = [0, 1]
    >>> Normal.param_limits
    {"mean": [-5, 5], "std": [0, 1]}

Now all instances of normally distributed columns will have mean between -5 and
5, and have standard deviation between 0 and 1.

They can be reset to their original limits by calling the :code:`reset` method::

    >>> Normal.reset()
    >>> Normal.param_limits
    {"mean": [-10, 10], "std": [0, 10]}

In addition to this, hard bounds on the parameters can be set::

    >>> Normal.hard_limits["mean"] = [-100, 100]

These hard limits are meant to stop the parameter limits from :ref:`shrinking
<compact>` too far.
