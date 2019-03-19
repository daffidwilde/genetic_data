""" .. A script containing all relevant functions for the crossover process. """

import numpy as np
import pandas as pd

from edo.individual import Individual

from .util import get_family_counts


def _collate_parents(parent1, parent2):
    """ Concatenate the columns and metadata from each parent together. These
    lists form a pool from which information is inherited during the crossover
    process. """

    parent_cols, parent_meta = [], []
    for dataframe, metadata in [parent1, parent2]:
        parent_cols += [dataframe[col] for col in dataframe.columns]
        parent_meta += metadata

    return parent_cols, parent_meta


def _cross_minimum_columns(parent_cols, parent_meta, col_limits, families):
    """ If :code:`col_limits` has a tuple lower limit, inherit the minimum
    number of columns from two parents to satisfy this limit. Return part of a
    whole individual and the adjusted parent information. """

    columns, metadata = [], []
    for limit, family in zip(col_limits[0], families):
        family_instances = [
            (col, pdf)
            for col, pdf in zip(parent_cols, parent_meta)
            if pdf.name == family.name
        ]

        for _ in range(limit):
            idx = np.random.choice(len(family_instances))
            col, pdf = family_instances.pop(idx)
            columns.append(col)
            metadata.append(pdf)

    return columns, metadata, parent_cols, parent_meta


def _cross_remaining_columns(
    columns, metadata, ncols, parent_cols, parent_meta, col_limits, families
):
    """ Regardless of whether :code:`col_limits` has a tuple upper limit or not,
    inherit all remaining columns from the two parents so as not to exceed these
    bounds. Return the components of a full individual. """

    family_counts = get_family_counts(metadata, families)
    while len(columns) < ncols:
        idx = np.random.choice(len(parent_cols))

        try:
            pdf = parent_meta[idx]
            family = pdf.family
            family_idx = families.index(family)
            if family_counts[family] < col_limits[1][family_idx]:
                columns.append(parent_cols.pop(idx))
                metadata.append(parent_meta.pop(idx))
                family_counts[family] += 1

        except TypeError:
            columns.append(parent_cols.pop(idx))
            metadata.append(parent_meta.pop(idx))

    return columns, metadata


def _adjust_column_lengths(columns, metadata, nrows):
    """ Trim or fill in the values of each column as needed. """

    idxs = None
    adjusted_columns = []
    for column, meta in zip(columns, metadata):
        difference = len(column) - nrows
        size = abs(difference)
        if difference > 0:
            if idxs is None:
                idxs = np.random.choice(len(column), size=size, replace=False)
            column = column.drop(idxs, axis=0).reset_index(drop=True)
        else:
            column = column.append(
                pd.Series(meta.sample(size)), ignore_index=False
            ).reset_index(drop=True)

        adjusted_columns.append(column)

    return adjusted_columns


def crossover(parent1, parent2, col_limits, families, prob=0.5):
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
    families : list
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

    parent_cols, parent_meta = _collate_parents(parent1, parent2)
    columns, metadata = [], []

    if np.random.random() < prob:
        nrows = len(parent1.dataframe)
    else:
        nrows = len(parent2.dataframe)

    if np.random.random() < prob:
        ncols = len(parent1.metadata)
    else:
        ncols = len(parent2.metadata)

    if isinstance(col_limits[0], tuple):
        columns, metadata, parent_cols, parent_meta = _cross_minimum_columns(
            parent_cols, parent_meta, col_limits, families
        )

    columns, metadata = _cross_remaining_columns(
        columns, metadata, ncols, parent_cols, parent_meta, col_limits, families
    )
    columns = _adjust_column_lengths(columns, metadata, nrows)

    dataframe = pd.DataFrame({i: col.values for i, col in enumerate(columns)})
    return Individual(dataframe, metadata)
