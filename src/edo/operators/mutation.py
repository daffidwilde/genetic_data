""" .. Functions related to the mutation operator. """

from edo.individual import Individual

from .util import get_family_counts


def mutation(individual, prob, row_limits, col_limits, families, weights=None):
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
    families: list
        Families of distributions with which to create new columns.
    weights : list, optional
        Probability with which to sample a distribution from :code:`families`.
        If :code:`None`, sample uniformly.

    Returns
    -------
    mutant : Individual
        A (potentially) mutated individual.
    """

    dataframe, metadata = individual
    random_state = individual.random_state
    dataframe, metadata = mutate_nrows(
        dataframe, metadata, row_limits, random_state, prob
    )
    dataframe, metadata = mutate_ncols(
        dataframe, metadata, col_limits, families, weights, random_state, prob
    )
    metadata = mutate_params(metadata, random_state, prob)

    dataframe = mutate_values(dataframe, metadata, random_state, prob)
    return Individual(dataframe, metadata, random_state)


def mutate_nrows(dataframe, metadata, row_limits, random_state, prob):
    """ Mutate the number of rows an individual has by adding a new row and/or
    dropping a row at random so as not to exceed the bounds of
    :code:`row_limits`. """

    if random_state.random() < prob and dataframe.shape[0] < row_limits[1]:
        dataframe = _add_row(dataframe, metadata, random_state)

    if random_state.random() < prob and dataframe.shape[0] > row_limits[0]:
        dataframe = _remove_row(dataframe, random_state)

    return dataframe, metadata


def mutate_ncols(
    dataframe, metadata, col_limits, families, weights, random_state, prob
):
    """ Mutate the number of columns an individual has by adding a new column
    and/or dropping a column at random. In either case, the bounds defined in
    :code:`col_limits` cannot be exceeded. """

    if isinstance(col_limits[1], tuple):
        condition = dataframe.shape[1] < sum(col_limits[1])
    else:
        condition = dataframe.shape[1] < col_limits[1]

    if random_state.random() < prob and condition:
        dataframe, metadata = _add_col(
            dataframe, metadata, col_limits, families, weights, random_state
        )

    if isinstance(col_limits[0], tuple):
        condition = dataframe.shape[1] > sum(col_limits[0])
    else:
        condition = dataframe.shape[1] > col_limits[0]

    if random_state.random() < prob and condition:
        dataframe, metadata = _remove_col(
            dataframe, metadata, col_limits, families, random_state
        )

    return dataframe, metadata


def mutate_params(metadata, random_state, prob):
    """ Mutate the parameters of each column in the metadata of an individual.
    Each mutation has probability :code:`prob`. """

    for pdf in metadata:
        subtype = pdf.__class__
        limits = pdf.param_limits
        for param in limits:
            if random_state.random() < prob:
                vars(pdf)[param] = vars(subtype(random_state))[param]

    return metadata


def mutate_values(dataframe, metadata, random_state, prob):
    """ Iterate over the values of :code:`dataframe`, mutating them with
    probability :code:`prob`. Mutation is done by resampling from each column's
    associated distribution in :code:`metadata`. """

    for j, col in enumerate(dataframe.columns):
        pdf = metadata[j]
        for i, value in enumerate(dataframe[col]):
            if random_state.random() < prob:
                value = pdf.sample(1, random_state)[0]
                dataframe.iloc[i, j] = value

    return dataframe


def _rename(dataframe):
    """ Rename columns or reindex to make sense after deletion or addition of a
    new line. """

    dataframe = dataframe.reset_index(drop=True)
    dataframe.columns = (i for i, _ in enumerate(dataframe.columns))
    return dataframe


def _add_row(dataframe, metadata, random_state):
    """ Append a row to the dataframe by sampling values from each column's
    distribution. """

    dataframe = dataframe.append(
        {i: pdf.sample(1, random_state)[0] for i, pdf in enumerate(metadata)},
        ignore_index=True,
    )

    return dataframe


def _remove_row(dataframe, random_state):
    """ Remove a row from a dataframe at random. """

    line = random_state.choice(dataframe.index)
    dataframe = _rename(dataframe.drop(line, axis=0))
    return dataframe


def _add_col(dataframe, metadata, col_limits, families, weights, random_state):
    """ Add a new column to the end of the dataframe by sampling a distribution
    from :code:`families` according to the column limits and distribution
    weights. """

    nrows, ncols = dataframe.shape
    if isinstance(col_limits[1], tuple):
        family_counts = get_family_counts(metadata, families)
        while len(dataframe.columns) != ncols + 1:
            family = random_state.choice(families, p=weights)
            idx = families.index(family)
            if family_counts[family] < col_limits[1][idx]:
                pdf = family.make_instance(random_state)
                dataframe[ncols] = pdf.sample(nrows, random_state)
                metadata.append(pdf)

        dataframe = _rename(dataframe)
        return dataframe, metadata

    family = random_state.choice(families, p=weights)
    pdf = family.make_instance(random_state)
    dataframe[ncols] = pdf.sample(nrows, random_state)
    metadata.append(pdf)

    dataframe = _rename(dataframe)
    return dataframe, metadata


def _remove_col(dataframe, metadata, col_limits, families, random_state):
    """ Remove a column (and its metadata) from a dataframe at random. """

    if isinstance(col_limits[0], tuple):
        ncols = dataframe.shape[1]
        family_counts = get_family_counts(metadata, families)
        while len(dataframe.columns) != ncols - 1:
            col = random_state.choice(dataframe.columns)
            idx = dataframe.columns.get_loc(col)
            pdf = metadata[idx]
            family = pdf.family
            family_idx = families.index(family)
            if family_counts[family] > col_limits[0][family_idx]:
                dataframe = _rename(dataframe.drop(col, axis=1))
                metadata.pop(idx)

        return dataframe, metadata

    col = random_state.choice(dataframe.columns)
    idx = dataframe.columns.get_loc(col)
    dataframe = _rename(dataframe.drop(col, axis=1))
    metadata.pop(idx)

    return dataframe, metadata
