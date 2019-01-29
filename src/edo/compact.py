""" Function for compacting the search space. """


def compact_search_space(parents, pdfs, itr, max_iter, compaction_ratio):
    """ Given the current progress of the GA, compact the search space, i.e. the
    parameter spaces for each of the distribution classes in :code:`pdfs`. """

    if compaction_ratio == 1:
        raise ValueError("Compaction ratio, s, must satisfy 0 <= s < 1.")

    if compaction_ratio == 0:
        return pdfs

    compact_factor = 1 - (itr / (compaction_ratio * max_iter))

    pass
