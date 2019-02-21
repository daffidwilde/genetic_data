""" .. A script containing all relevant functions for the crossover process. """

import numpy as np
import pandas as pd

from edo.individual import Individual

from .util import _get_pdf_counts


def _collate_parents(parent1, parent2):
    """ Concatenate the columns and metadata from each parent together. These
    lists form a pool from which information is inherited during the crossover
    process. """

    parent_cols, parent_metadata = [], []
    for dataframe, metadata in [parent1, parent2]:
        parent_cols += [dataframe[col] for col in dataframe.columns]
        parent_metadata += metadata

    return parent_cols, parent_metadata


def _cross_minimum_columns(parent_cols, parent_metadata, col_limits, pdfs):
    """ If :code:`col_limits` has a tuple lower limit, inherit the minimum
    number of columns from two parents to satisfy this limit. Return part of a
    whole individual and the adjusted parent information. """

    cols, metadata = [], []
    for limit, pdf_class in zip(col_limits[0], pdfs):
        pdf_instances = [
            (col, pdf)
            for col, pdf in zip(parent_cols, parent_metadata)
            if isinstance(pdf, pdf_class)
        ]

        for _ in range(limit):
            idx = np.random.choice(len(pdf_instances))
            col, pdf = pdf_instances.pop(idx)
            cols.append(col)
            metadata.append(pdf)

    return cols, metadata, parent_cols, parent_metadata


def _cross_remaining_columns(
    cols, metadata, ncols, parent_cols, parent_metadata, col_limits, pdfs
):
    """ Regardless of whether :code:`col_limits` has a tuple upper limit or not,
    inherit all remaining columns from the two parents so as not to exceed these
    bounds. Return the components of a full inidividual. """

    pdf_counts = _get_pdf_counts(metadata, pdfs)
    while len(cols) < ncols:
        idx = np.random.choice(len(parent_cols))
        pdf = parent_metadata[idx]
        pdf_class = pdf.__class__
        pdf_class_idx = pdfs.index(pdf_class)

        try:
            if pdf_counts[pdf_class] < col_limits[1][pdf_class_idx]:
                cols.append(parent_cols.pop(idx))
                metadata.append(parent_metadata.pop(idx))
                pdf_counts[pdf_class] += 1

        except TypeError:
            cols.append(parent_cols.pop(idx))
            metadata.append(parent_metadata.pop(idx))

    return cols, metadata


def _adjust_column_lengths(cols, metadata, nrows):
    """ Trim or fill in the values of each column as needed. """

    adjusted_cols = []
    for col, pdf in zip(cols, metadata):
        difference = len(col) - nrows
        size = abs(difference)

        if difference > 0:
            idxs = np.random.choice(col.index, size=size, replace=False)
            col = col.drop(idxs, axis=0).reset_index(drop=True)
        elif difference < 0:
            col = col.append(
                pd.Series(pdf.sample(size)), ignore_index=False
            ).reset_index(drop=True)

        adjusted_cols.append(col)

    return adjusted_cols


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

    parent_cols, parent_metadata = _collate_parents(parent1, parent2)
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
        cols, metadata, parent_cols, parent_metadata = _cross_minimum_columns(
            parent_cols, parent_metadata, col_limits, pdfs
        )

    cols, metadata = _cross_remaining_columns(
        cols, metadata, ncols, parent_cols, parent_metadata, col_limits, pdfs
    )
    cols = _adjust_column_lengths(cols, metadata, nrows)

    dataframe = pd.DataFrame({i: col.values for i, col in enumerate(cols)})
    return Individual(dataframe, metadata)
