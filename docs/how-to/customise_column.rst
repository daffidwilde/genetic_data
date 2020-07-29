Customise column distributions
------------------------------

All distributions in ``edo`` have settings that can be customised. You can see
all of the currently implemented distributions, and their default settings, on
the :mod:`edo.distributions` reference page. For now, let's consider the
normal distribution::

    >>> from edo.distributions import Normal

The default bounds are -10 and 10 for the mean, and 0 and 10 for the standard
deviation::

    >>> Normal.param_limits
    {'mean': [-10, 10], 'std': [0, 10]}

Changing these bounds is as simple as redefining the class attributes::

    >>> Normal.param_limits['mean'] = [-5, 5]
    >>> Normal.param_limits['std'] = [0, 1]
    >>> Normal.param_limits
    {'mean': [-5, 5], 'std': [0, 1]}

Now all instances of normally distributed columns will have a mean between -5
and 5, and a standard deviation between 0 and 1.

In addition to this, hard bounds on the parameters can be set::

    >>> Normal.hard_limits['mean'] = [-100, 100]

These hard limits are meant to stop the parameter limits from :ref:`shrinking
<compact>` too far.
