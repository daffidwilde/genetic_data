""" Functions for shrinking the search space. """


def _get_param_values(parents, pdf, name):
    """ Get the values of a distribution present amongst all parents. """

    values = []
    for _, metadata in parents:
        for column in metadata:
            if isinstance(column, pdf):
                try:
                    for val in vars(column)[name]:
                        values.append(val)
                except TypeError:
                    values.append(vars(column)[name])

    return values


def _adjust_pdf_params(parents, pdf, itr, shrinkage):
    r""" Adjust the search space of a distribution's parameters according to a
    power law on its limits:

    .. math::

        u_{t+1} - l_{t+1} = (u_t - l_t) * k^t,

    where :math:`t` is the current timestep, :math:`u_t, l_t` denote the upper
    and lower limits of the parameter at iteration :math:`t`, and :math:`k \in
    (0, 1)` is some :code:`shrinkage` factor.
    """

    for name, limits in pdf.param_limits.items():
        values = _get_param_values(parents, pdf, name)
        if values:
            midpoint = sum(values) / len(values)
            shift = (max(limits) - min(limits)) * (shrinkage ** itr) / 2

            lower = max(min(limits), midpoint - shift)
            upper = min(min(limits), midpoint + shift)

            pdf.param_limits[name] = sorted([lower, upper])

    return pdf


def shrink(parents, families, itr, shrinkage):
    """ Given the current progress of the evolutionary algorithm, shrink its
    search space, i.e. the parameter spaces for each of the distribution classes
    in :code:`families`.

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
        for pdf in family.subtypes:
            pdf = _adjust_pdf_params(parents, pdf, itr, shrinkage)

    return families
