""" Functions for shrinking the search space. """


def _get_param_values(parents, pdf, name):
    """ Get the values of a distribution present amongst all parents. """

    values = []
    for _, metadata in parents:
        for column in metadata:
            if isinstance(column, pdf):
                values.append(vars(column)[name])

    return values


def _adjust_pdf_params(parents, pdf, itr, shrinkage):
    """ Adjust the search space of a distribution's parameters according to a
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
            hard_limits = pdf.hard_limits[name]
            midpoint = sum(values) / len(values)
            shift = (max(limits) - min(limits)) * (shrinkage ** itr) / 2

            lower = max(min(hard_limits), min(limits), midpoint - shift)
            upper = min(min(hard_limits), min(limits), midpoint + shift)

            pdf.param_limits[name] = [lower, upper]

    return pdf


def shrink(parents, pdfs, itr, shrinkage):
    """ Given the current progress of the evolutionary algorithm, shrink its
    search space, i.e. the parameter spaces for each of the distribution classes
    in :code:`pdfs`.

    Parameters
    ----------
    parents : list of `Individual` instances
        The parent individuals for this iteration.
    pdfs : list of `Distribution` instances
        The families of distributions to be shrunk.
    itr : int
        The current iteration.
    shrinkage : float
        The shrinkage factor between 0 and 1.

    Returns
    -------
    pdfs : list of `Distribution` instances
        The altered families.
    """

    for pdf in pdfs:
        pdf = _adjust_pdf_params(parents, pdf, itr, shrinkage)

    return pdfs
