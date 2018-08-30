""" Functions related to the mutation operator. """

from copy import deepcopy

import numpy as np

from ..individual import Individual
from .util import _add_line, _remove_line


def _mutate_nrows(dataframe, metadata, row_limits, prob):
    """ Mutate the number of rows an individual has by adding a new row and/or
    dropping a row at random so as not to exceed the bounds of
    :code:`row_limits`. """

    if np.random.random() < prob and dataframe.shape[0] < row_limits[1]:
        dataframe, metadata = _add_line(dataframe, metadata, axis=0)

    if np.random.random() < prob and dataframe.shape[0] > row_limits[0]:
        dataframe, metadata = _remove_line(dataframe, metadata, axis=0)

    return dataframe, metadata


def _mutate_ncols(dataframe, metadata, col_limits, pdfs, weights, prob):
    """ Mutate the number of columns an individual has by adding a new column
    and/or dropping a column at random. In either case, the bounds defined in
    :code:`col_limits` cannot be exceeded. """

    if isinstance(col_limits[1], tuple):
        condition = dataframe.shape[1] < sum(col_limits[1])
    else:
        condition = dataframe.shape[1] < col_limits[1]

    if np.random.random() < prob and condition:
        dataframe, metadata = _add_line(
            dataframe, metadata, 1, col_limits, pdfs, weights
        )

    if isinstance(col_limits[0], tuple):
        condition = dataframe.shape[1] > sum(col_limits[0])
    else:
        condition = dataframe.shape[1] > col_limits[0]

    if np.random.random() < prob and condition:
        dataframe, metadata = _remove_line(
            dataframe, metadata, 1, col_limits, pdfs
        )

    return dataframe, metadata


def _mutate_values(dataframe, metadata, prob):
    """ Iterate over the values of :code:`dataframe`, mutating them with
    probability :code:`prob`. Mutation is done by resampling from each column's
    associated distribution in :code:`metadata`. """

    for j, col in enumerate(dataframe.columns):
        pdf = metadata[j]
        for i, value in enumerate(dataframe[col]):
            if np.random.random() < prob:
                value = pdf.sample(1)[0]
                dataframe.iloc[i, j] = value

    return dataframe


def mutation(individual, prob, row_limits, col_limits, pdfs, weights):
    """ Mutate an individual. Here, the characteristics of an individual can be
    split into two parts: their dimensions, and their values. Each of these
    parts is mutated in a different way using the same probability, `prob`. """

    dataframe, metadata = deepcopy(individual)
    dataframe, metadata = _mutate_nrows(dataframe, metadata, row_limits, prob)
    dataframe, metadata = _mutate_ncols(
        dataframe, metadata, col_limits, pdfs, weights, prob
    )

    dataframe = _mutate_values(dataframe, metadata, prob)
    return Individual(dataframe, metadata)
