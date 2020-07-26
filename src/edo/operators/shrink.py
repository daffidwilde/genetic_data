""" Functions for shrinking the search space. """


def _get_param_values(parents, subtype, name):
    """ Get the parameter values from the parents for a particular distribution
    subtype parameter. """

    values = []
    for _, metadata in parents:
        for pdf in metadata:
            if isinstance(pdf, subtype):
                try:
                    for val in vars(pdf)[name]:
                        values.append(val)
                except TypeError:
                    values.append(vars(pdf)[name])

    return values


def _adjust_subtype_param_limits(parents, subtype, itr, shrinkage):
    r""" Adjust the search space of a distribution subtype's parameters
    according to a power law on its limits:

    .. math::

        u_{t+1} - l_{t+1} = (u_t - l_t) * k^t,

    where :math:`t` is the current timestep, :math:`u_t, l_t` denote the upper
    and lower limits of the parameter at iteration :math:`t`, and :math:`k \in
    (0, 1)` is some ``shrinkage`` factor.
    """

    for name, limits in subtype.param_limits.items():
        values = _get_param_values(parents, subtype, name)
        if values:
            midpoint = sum(values) / len(values)
            shift = (max(limits) - min(limits)) * (shrinkage ** itr) / 2

            lower = max(min(limits), midpoint - shift)
            upper = min(min(limits), midpoint + shift)

            subtype.param_limits[name] = sorted([lower, upper])

    return subtype


def shrink(parents, families, itr, shrinkage):
    """ Given the current progress of the evolutionary algorithm, shrink its
    search space, i.e. the parameter spaces for each of the distribution classes
    in ``families``.

    Parameters
    ----------
    parents : list of `Individual` instances
        The parent individuals for this iteration.
    families : list of `Distribution` instances
        The families of distributions to be shrunk.
    itr : int
        The current iteration.
    shrinkage : float
        The shrinkage factor between 0 and 1.

    Returns
    -------
    families : list of `Distribution` instances
        The altered families.
    """

    for family in families:
        for i, subtype in family.subtypes.items():
            subtype = _adjust_subtype_param_limits(
                parents, subtype, itr, shrinkage
            )
            family.subtypes[i] = subtype

    return families
