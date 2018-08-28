""" .. A script containing all relevant functions for the crossover process. """

import numpy as np
import pandas as pd

from ..individual import Individual
from .util import _add_line, _fillna, _get_pdf_counts


def _collate_parents(parent1, parent2):
    """ Concatenate the columns and metadata from each parent together. These
    lists form a pool from which information is inherited during the crossover
    process. """

    parent_columns, parent_metadata = [], []
    for dataframe, metadata in [parent1, parent2]:
        parent_columns += [dataframe[col] for col in dataframe.columns]
        parent_metadata += metadata

    return parent_columns, parent_metadata


def _cross_minimum_cols(parent_columns, parent_metadata, col_limits, pdfs):
    """ If :code:`col_limits` has a tuple lower limit, inherit the minimum
    number of columns from two parents to satisfy this limit. Return part of a
    whole individual and the adjusted parent information. """

    all_idxs, cols, metadata = [], [], []
    for limit, pdf_class in zip(col_limits[0], pdfs):
        if limit:
            pdf_class_idxs = np.where(
                [isinstance(pdf, pdf_class) for pdf in parent_metadata]
            )

            idxs = np.random.choice(*pdf_class_idxs, size=limit, replace=False)
            for idx in idxs:
                metadata.append(parent_metadata[idx])
                cols.append(parent_columns[idx])
                all_idxs.append(idx)

    for idx in sorted(all_idxs, reverse=True):
        parent_columns.pop(idx)
        parent_metadata.pop(idx)

    return cols, metadata, parent_columns, parent_metadata


def _cross_remaining_cols(
    cols, metadata, ncols, parent_columns, parent_metadata, col_limits, pdfs
):
    """ Regardless of whether :code:`col_limits` has a tuple upper limit or not,
    inherit all remaining columns from the two parents so as not to exceed these
    bounds. Return the components of a full inidividual. """

    pdf_counts = _get_pdf_counts(metadata, pdfs)
    while len(cols) < ncols:
        idx = np.random.randint(len(parent_columns))
        pdf = parent_metadata[idx]
        pdf_idx = pdfs.index(pdf.__class__)

        try:
            if pdf_counts[pdf.__class__] < col_limits[1][pdf_idx]:
                cols.append(parent_columns.pop(idx))
                metadata.append(parent_metadata.pop(idx))
                pdf_counts[pdf.__class__] += 1

        except TypeError:
            cols.append(parent_columns.pop(idx))
            metadata.append(parent_metadata.pop(idx))

    return cols, metadata


def crossover(parent1, parent2, col_limits, pdfs, prob=0.5):
    """ Blend the information from two parents to create a new
    :code:`Individual`. Dimensions are inherited first, and then column-metadata
    pairs are inherited from either parent uniformly. Missing values are filled
    in as necessary.

    Parameters
    ----------
    parent1 : Individual
        The first individual to be blended.
    parent2 : Individual
        The second individual to be blended.
    col_limits : list
        Lower and upper bounds on the number of columns :code:`offspring` can
        have. Used in case of tuple limits.
    pdfs : list
        Families of distributions with which to create new columns. Used in case
        of tuple column limits.
    prob : float, optional
        The cut-off probability with which to inherit dimensions from
        :code:`parent1` over :code:`parent2`.

    Returns
    -------
    offspring : Individual
        A new individual formed from the dimensions and columns of its parents.
    """

    parent_columns, parent_metadata = _collate_parents(parent1, parent2)
    cols, metadata = [], []

    if np.random.random() < prob:
        nrows = len(parent1.dataframe)
    else:
        nrows = len(parent2.dataframe)

    if np.random.random() < prob:
        ncols = len(parent1.metadata)
    else:
        ncols = len(parent2.metadata)

    if isinstance(col_limits[0], tuple):
        cols, metadata, parent_columns, parent_metadata = _cross_minimum_cols(
            parent_columns, parent_metadata, col_limits, pdfs
        )

    cols, metadata = _cross_remaining_cols(
        cols, metadata, ncols, parent_columns, parent_metadata, col_limits, pdfs
    )

    dataframe = pd.DataFrame({i: col for i, col in enumerate(cols)})

    while len(dataframe) != nrows:
        if len(dataframe) > nrows:
            dataframe = dataframe.iloc[:nrows, :]
        elif len(dataframe) < nrows:
            dataframe, metadata = _add_line(dataframe, metadata, axis=0)

    dataframe = _fillna(dataframe, metadata)
    return Individual(dataframe, metadata)
