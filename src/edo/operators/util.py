""" A collection of useful functions for various processes within the GA. """

import numpy as np


def _get_pdf_counts(metadata, pdfs):
    """ Get the count of each pdf class present in metadata. """

    return {
        pdf_class: sum([isinstance(pdf, pdf_class) for pdf in metadata])
        for pdf_class in pdfs
    }


def _rename(dataframe):
    """ Rename metadata or reindex to make sense after deletion or addition of a
    new line. """

    dataframe = dataframe.reset_index(drop=True)
    dataframe.columns = range(len(dataframe.columns))
    return dataframe


def _remove_row(dataframe):
    """ Remove a row from a dataframe at random. """

    line = np.random.choice(dataframe.index)
    dataframe = _rename(dataframe.drop(line, axis=0))
    return dataframe


def _remove_col(dataframe, metadata, col_limits, pdfs):
    """ Remove a column (and its metadata) from a dataframe at random. """

    if isinstance(col_limits[0], tuple):
        ncols = dataframe.shape[1]
        pdf_counts = _get_pdf_counts(metadata, pdfs)
        while len(dataframe.columns) != ncols - 1:
            col = np.random.choice(dataframe.columns)
            col_idx = dataframe.columns.get_loc(col)
            pdf = metadata[col_idx]
            pdf_class = pdf.__class__
            pdf_idx = pdfs.index(pdf_class)
            if pdf_counts[pdf_class] > col_limits[0][pdf_idx]:
                dataframe = _rename(dataframe.drop(col, axis=1))
                metadata.pop(col_idx)

        return dataframe, metadata

    col = np.random.choice(dataframe.columns)
    idx = dataframe.columns.get_loc(col)
    dataframe = _rename(dataframe.drop(col, axis=1))
    metadata.pop(idx)

    return dataframe, metadata


def _add_row(dataframe, metadata):
    """ Append a row to the dataframe by sampling values from each column's
    distribution. """

    dataframe = dataframe.append(
        {i: pdf.sample(1)[0] for i, pdf in enumerate(metadata)},
        ignore_index=True,
    )

    return dataframe


def _add_col(dataframe, metadata, col_limits, pdfs, weights):
    """ Add a new column to the end of the dataframe by sampling a distribution
    from :code:`pdfs` according to the column limits and distribution weights.
    """

    nrows, ncols = dataframe.shape
    if isinstance(col_limits[1], tuple):
        pdf_counts = _get_pdf_counts(metadata, pdfs)
        while len(dataframe.columns) != ncols + 1:
            pdf_class = np.random.choice(pdfs, p=weights)
            idx = pdfs.index(pdf_class)
            if pdf_counts[pdf_class] < col_limits[1][idx]:
                pdf = pdf_class()
                dataframe[ncols] = pdf.sample(nrows)
                metadata.append(pdf)

        dataframe = _rename(dataframe)
        return dataframe, metadata

    pdf_class = np.random.choice(pdfs, p=weights)
    pdf = pdf_class()
    dataframe[ncols] = pdf.sample(nrows)
    metadata.append(pdf)

    dataframe = _rename(dataframe)
    return dataframe, metadata
