""" Function for compacting the search space. """


def _get_all_param_values(parents):
    """ Get all the parent parameter values present in a dictionary. """

    all_param_vals = {}
    for _, metadata in parents:
        for column in metadata:
            col = column.__class__
            if col not in all_param_vals.keys():
                all_param_vals[col] = {}
            for param_name, param_val in column.__dict__.items():
                try:
                    all_param_vals[col][param_name] += [param_val]
                except KeyError:
                    all_param_vals[col][param_name] = [param_val]

    return all_param_vals


def compact_search_space(parents, pdfs, itr, compaction_ratio):
    """ Given the current progress of the GA, compact the search space, i.e. the
    parameter spaces for each of the distribution classes in :code:`pdfs`. """

    all_param_vals = _get_all_param_values(parents)
    for pdf, params in all_param_vals.items():
        for name, vals in params.items():

            limits = pdf.param_limits[name]
            hard_limits = pdf.hard_limits[name]

            midpoint = sum(vals) / len(vals)
            shift = (max(limits) - min(limits)) * (compaction_ratio ** itr) / 2

            lower = max(
                (hard_limits[0], min(limits), midpoint - shift)
            )
            upper = min(
                (hard_limits[1], max(limits), midpoint + shift)
            )

            pdf.param_limits[name] = [lower, upper]

    return pdfs
