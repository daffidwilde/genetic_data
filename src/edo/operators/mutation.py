""" .. Functions related to the mutation operator. """

import numpy as np

from edo.individual import Individual

from .util import get_family_counts


def mutation(individual, prob, row_limits, col_limits, pdfs, weights=None):
    """ Mutate an individual. Here, the characteristics of an individual can be
    split into two parts: their dimensions, and their values. Each of these
    parts is mutated in a different way using the same probability,
    :code:`prob`.

    Parameters
    ----------
    individual : Individual
        The individual to be mutated.
    prob : float
        The probability with which any characteristic of :code:`individual`
        should be mutated.
    row_limits : list
        Lower and upper limits on the number of rows an individual can have.
    col_limits : list
        Lower and upper limits on the number of columns an individual can have.
    pdfs : list
        Families of distributions with which to create new columns.
    weights : list, optional
        Probability with which to sample a distribution from :code:`pdfs`. If
        :code:`None`, sample uniformly.

    Returns
    -------
    mutant : Individual
        A (potentially) mutated individual.
    """

    dataframe, metadata = individual
    dataframe, metadata = mutate_nrows(dataframe, metadata, row_limits, prob)
    dataframe, metadata = mutate_ncols(
        dataframe, metadata, col_limits, pdfs, weights, prob
    )
    metadata = mutate_params(metadata, prob)

    dataframe = mutate_values(dataframe, metadata, prob)
    return Individual(dataframe, metadata)


def mutate_nrows(dataframe, metadata, row_limits, prob):
    """ Mutate the number of rows an individual has by adding a new row and/or
    dropping a row at random so as not to exceed the bounds of
    :code:`row_limits`. """

    if np.random.random() < prob and dataframe.shape[0] < row_limits[1]:
        dataframe = _add_row(dataframe, metadata)

    if np.random.random() < prob and dataframe.shape[0] > row_limits[0]:
        dataframe = _remove_row(dataframe)

    return dataframe, metadata


def mutate_ncols(dataframe, metadata, col_limits, pdfs, weights, prob):
    """ Mutate the number of columns an individual has by adding a new column
    and/or dropping a column at random. In either case, the bounds defined in
    :code:`col_limits` cannot be exceeded. """

    if isinstance(col_limits[1], tuple):
        condition = dataframe.shape[1] < sum(col_limits[1])
    else:
        condition = dataframe.shape[1] < col_limits[1]

    if np.random.random() < prob and condition:
        dataframe, metadata = _add_col(
            dataframe, metadata, col_limits, pdfs, weights
        )

    if isinstance(col_limits[0], tuple):
        condition = dataframe.shape[1] > sum(col_limits[0])
    else:
        condition = dataframe.shape[1] > col_limits[0]

    if np.random.random() < prob and condition:
        dataframe, metadata = _remove_col(dataframe, metadata, col_limits, pdfs)

    return dataframe, metadata


def mutate_params(metadata, prob):
    """ Mutate the parameters of each column in the metadata of an individual.
    Each mutation has probability :code:`prob`. """

    for pdf in metadata:
        pdf_class = pdf.__class__
        limits = pdf.param_limits
        for param in limits:
            if np.random.random() < prob:
                vars(pdf)[param] = vars(pdf_class())[param]

    return metadata


def mutate_values(dataframe, metadata, prob):
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


def _rename(dataframe):
    """ Rename columns or reindex to make sense after deletion or addition of a
    new line. """

    dataframe = dataframe.reset_index(drop=True)
    dataframe.columns = (i for i, _ in enumerate(dataframe.columns))
    return dataframe


def _add_row(dataframe, metadata):
    """ Append a row to the dataframe by sampling values from each column's
    distribution. """

    dataframe = dataframe.append(
        {i: pdf.sample(1)[0] for i, pdf in enumerate(metadata)},
        ignore_index=True,
    )

    return dataframe


def _remove_row(dataframe):
    """ Remove a row from a dataframe at random. """

    line = np.random.choice(dataframe.index)
    dataframe = _rename(dataframe.drop(line, axis=0))
    return dataframe


def _add_col(dataframe, metadata, col_limits, families, weights):
    """ Add a new column to the end of the dataframe by sampling a distribution
    from :code:`families` according to the column limits and distribution weights.
    """

    nrows, ncols = dataframe.shape
    if isinstance(col_limits[1], tuple):
        family_counts = get_family_counts(metadata, families)
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


def _remove_col(dataframe, metadata, col_limits, families):
    """ Remove a column (and its metadata) from a dataframe at random. """

    if isinstance(col_limits[0], tuple):
        ncols = dataframe.shape[1]
        family_counts = get_family_counts(metadata, families)
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
