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


def compact_search_space(parents, pdfs, itr, max_iter, compaction_ratio):
    """ Given the current progress of the GA, compact the search space, i.e. the
    parameter spaces for each of the distribution classes in :code:`pdfs`. """

    if (
        compaction_ratio in [0, 1]
        or compaction_ratio < 0
        or compaction_ratio > 1
    ):
        raise ValueError("Compaction ratio, s, must satisfy 0 < s < 1.")

    compact_factor = 1 - (itr / (compaction_ratio * max_iter))
    all_param_vals = _get_all_param_values(parents)

    for pdf, params in all_param_vals.items():
        for name, vals in params.items():

            midpoint = sum(vals) / len(vals)
            shift = (max(vals) - min(vals)) * compact_factor / 2

            lower = midpoint - shift
            upper = midpoint + shift

            if shift < 0:
                lower, upper = upper, lower

            lower = max([pdf.hard_limits[name][0], lower])
            upper = min([pdf.hard_limits[name][1], upper])

            pdf.param_limits[name] = [lower, upper]

    return pdfs
