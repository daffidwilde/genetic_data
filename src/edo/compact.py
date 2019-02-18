""" Function for compacting the search space. """


def _get_param_values(parents, pdf, name):
    """ Get the values of a distribution present amongst all parents. """

    values = []
    for _, metadata in parents:
        for column in metadata:
            if isinstance(column, pdf):
                values.append(vars(column)[name])

    return values


def _adjust_pdf_params(parents, pdf, itr, shrinkage):
    """ Adjust the search space of a distribution's parameters. """

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


def compact_search_space(parents, pdfs, itr, shrinkage):
    """ Given the current progress of the GA, compact the search space, i.e. the
    parameter spaces for each of the distribution classes in :code:`pdfs`. """

    for pdf in pdfs:
        pdf = _adjust_pdf_params(parents, pdf, itr, shrinkage)

    return pdfs
