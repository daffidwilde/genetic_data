""" A collection of useful functions for various processes within the GA. """

import numpy as np


def _get_pdf_counts(metadata, families):
    """ Get the count of each pdf class present in metadata. """

    return {
        family: sum([pdf.name == family.name for pdf in metadata])
        for family in families
    }


def _rename(dataframe):
    """ Rename columns or reindex to make sense after deletion or addition of a
    new line. """

    dataframe = dataframe.reset_index(drop=True)
    dataframe.columns = (i for i, _ in enumerate(dataframe.columns))
    return dataframe


def _remove_row(dataframe):
    """ Remove a row from a dataframe at random. """

    line = np.random.choice(dataframe.index)
    dataframe = _rename(dataframe.drop(line, axis=0))
    return dataframe


def _remove_col(dataframe, metadata, col_limits, families):
    """ Remove a column (and its metadata) from a dataframe at random. """

    if isinstance(col_limits[0], tuple):
        ncols = dataframe.shape[1]
        family_counts = _get_pdf_counts(metadata, families)
        while len(dataframe.columns) != ncols - 1:
            col = np.random.choice(dataframe.columns)
            idx = dataframe.columns.get_loc(col)
            pdf = metadata[idx]
            family = pdf.family
            family_idx = families.index(family)
            if family_counts[family] > col_limits[0][family_idx]:
                dataframe = _rename(dataframe.drop(col, axis=1))
                metadata.pop(idx)

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


def _add_col(dataframe, metadata, col_limits, families, weights):
    """ Add a new column to the end of the dataframe by sampling a distribution
    from :code:`families` according to the column limits and distribution weights.
    """

    nrows, ncols = dataframe.shape
    if isinstance(col_limits[1], tuple):
        family_counts = _get_pdf_counts(metadata, families)
        while len(dataframe.columns) != ncols + 1:
            family = np.random.choice(families, p=weights)
            idx = families.index(family)
            if family_counts[family] < col_limits[1][idx]:
                pdf = family.make_instance()
                dataframe[ncols] = pdf.sample(nrows)
                metadata.append(pdf)

        dataframe = _rename(dataframe)
        return dataframe, metadata

    family = np.random.choice(families, p=weights)
    pdf = family.make_instance()
    dataframe[ncols] = pdf.sample(nrows)
    metadata.append(pdf)

    dataframe = _rename(dataframe)
    return dataframe, metadata
