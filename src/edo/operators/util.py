""" A collection of useful functions for various processes within the GA. """

import numpy as np


def _get_pdf_counts(metadata, pdfs):
    """ Get the count of each pdf class present in metadata. """

    return {
        pdf_class: sum([isinstance(pdf, pdf_class) for pdf in metadata])
        for pdf_class in pdfs
    }


def _rename(dataframe, axis):
    """ Rename metadata or reindex to make sense after deletion or addition of a
    new line. """

    if axis == 0:
        dataframe = dataframe.reset_index(drop=True)
    else:
        dataframe.columns = range(len(dataframe.columns))

    return dataframe


def _fillna(dataframe, metadata):
    """ Fill in `NaN` values of a column by sampling from the distribution
    associated with it. """

    for col, pdf in zip(dataframe.columns, metadata):
        data = dataframe[col]
        if data.isnull().any():
            nulls = data.isnull()
            samples = pdf.sample(nulls.sum())
            dataframe.loc[nulls, col] = samples

        dataframe[col] = dataframe[col].astype(pdf.dtype)

    return dataframe


def _remove_line(dataframe, metadata, axis, col_limits=None, pdfs=None):
    """ Remove a line (row or column) from a dataset at random. """

    if axis == 0:
        line = np.random.choice(dataframe.index)
        dataframe = _rename(dataframe.drop(line, axis=axis), axis)

    elif isinstance(col_limits[0], tuple):
        ncols = dataframe.shape[1]
        pdf_counts = _get_pdf_counts(metadata, pdfs)
        while len(dataframe.columns) != ncols - 1:
            line = np.random.choice(dataframe.columns)
            column_idx = dataframe.columns.get_loc(line)
            pdf = metadata[column_idx]
            pdf_class = pdf.__class__
            pdf_idx = pdfs.index(pdf_class)
            if pdf_counts[pdf_class] > col_limits[0][pdf_idx]:
                dataframe = _rename(dataframe.drop(line, axis=axis), axis)
                metadata.pop(column_idx)

    else:
        line = np.random.choice(dataframe.columns)
        idx = dataframe.columns.get_loc(line)
        dataframe = _rename(dataframe.drop(line, axis=axis), axis)
        metadata.pop(idx)

    return dataframe, metadata


def _add_line(
    dataframe, metadata, axis, col_limits=None, pdfs=None, weights=None
):
    """ Add a line (row or column) to the end of a dataset. Rows are added by
    sampling from the distribution associated with that column in `metadata`.
    metadata are added in the same way that they are at the initial creation of
    an individual by sampling from the list of all `pdfs` according to
    `weights`. """

    nrows, ncols = dataframe.shape

    if axis == 0:
        dataframe = dataframe.append(
            {i: pdf.sample(1)[0] for i, pdf in enumerate(metadata)},
            ignore_index=True,
        )

    elif isinstance(col_limits[1], tuple):
        pdf_counts = _get_pdf_counts(metadata, pdfs)
        while len(dataframe.columns) != ncols + 1:
            pdf_class = np.random.choice(pdfs, p=weights)
            idx = pdfs.index(pdf_class)
            pdf = pdf_class()
            if pdf_counts[pdf_class] < col_limits[1][idx]:
                dataframe[ncols + 1] = pdf.sample(nrows)
                metadata.append(pdf)
    else:
        pdf_class = np.random.choice(pdfs, p=weights)
        pdf = pdf_class()
        dataframe[ncols + 1] = pdf.sample(nrows)
        metadata.append(pdf)

    dataframe = _rename(dataframe, axis)
    return dataframe, metadata
