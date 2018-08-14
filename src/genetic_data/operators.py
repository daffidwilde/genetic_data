""" Default crossover and mutation operators for a genetic algorithm. """

from copy import deepcopy

import numpy as np
import pandas as pd

from genetic_data.individual import Individual


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
        dataframe.columns = [f"col_{i}" for i in range(len(dataframe.columns))]

    return dataframe


def _fillna(dataframe, metadata):
    """ Fill in `NaN` values of a column by sampling from the distribution
    associated with it. """

    for i, col in enumerate(dataframe.columns):
        data = dataframe[col]
        pdf = metadata[i]
        if data.isnull().any():
            nulls = data.isnull()
            samples = pdf.sample(nulls.sum())
            dataframe.loc[nulls, col] = samples

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
            {f"col_{i}": pdf.sample(1)[0] for i, pdf in enumerate(metadata)},
            ignore_index=True,
        )

    elif isinstance(col_limits[1], tuple):
        pdf_counts = _get_pdf_counts(metadata, pdfs)
        while len(dataframe.columns) != ncols + 1:
            pdf_class = np.random.choice(pdfs, p=weights)
            idx = pdfs.index(pdf_class)
            pdf = pdf_class()
            if pdf_counts[pdf_class] < col_limits[1][idx]:
                dataframe[f"col_{ncols + 1}"] = pdf.sample(nrows)
                metadata.append(pdf)
    else:
        pdf_class = np.random.choice(pdfs, p=weights)
        pdf = pdf_class()
        dataframe[f"col_{ncols + 1}"] = pdf.sample(nrows)
        metadata.append(pdf)

    dataframe = _rename(dataframe, axis)
    return dataframe, metadata


def get_fitness(fitness, population):
    """ Return the fitness score of each individual in a population. """

    return [fitness(individual.dataframe) for individual in population]


def selection(population, pop_fitness, best_prop, lucky_prop, maximise):
    """ Given a population, select a proportion of the `best` individuals and
    another of the `lucky` individuals (if they are available) to form a set of
    potential parents. This mirrors the survival of the fittest paradigm whilst
    including a number of less-fit individuals to stop the algorithm from
    converging too early on a suboptimal population. """

    size = len(population)
    num_best = int(best_prop * size)
    num_lucky = int(lucky_prop * size)

    if maximise:
        best_choice = np.argmax
    else:
        best_choice = np.argmin

    if num_best == 0 and num_lucky == 0:
        raise ValueError(
            'Not a large enough proportion of "best" and/or \
                          "lucky" individuals chosen. Reconsider these values.'
        )

    population = deepcopy(population)
    pop_fitness = deepcopy(pop_fitness)
    parents = []
    for _ in range(num_best):
        if population != []:
            best = best_choice(pop_fitness)
            pop_fitness.pop(best)
            parents.append(population.pop(best))

    for _ in range(num_lucky):
        if population != []:
            lucky = np.random.choice(len(population))
            parents.append(population.pop(lucky))

    return parents


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


def crossover(parent1, parent2, col_limits, pdfs):
    """ Blend the information from two parents to create a new
    :code:`Individual`. Dimensions are inherited equally from either parent,
    then column-metadata pairs from each parent are pooled together and sampled
    uniformly according to :code:`col_limits`. This information is then collated
    to form a new individual, filling in missing values as necessary. """

    parent_columns, parent_metadata = _collate_parents(parent1, parent2)
    cols, metadata = [], []

    if np.random.random() < 0.5:
        nrows = len(parent1.dataframe)
    else:
        nrows = len(parent2.dataframe)

    if np.random.random() < 0.5:
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

    dataframe = pd.DataFrame({f"col_{i}": col for i, col in enumerate(cols)})

    while len(dataframe) != nrows:
        if len(dataframe) > nrows:
            dataframe = dataframe.iloc[:nrows, :]
        elif len(dataframe) < nrows:
            dataframe, metadata = _add_line(dataframe, metadata, axis=0)

    dataframe = _fillna(dataframe, metadata)
    return Individual(dataframe, metadata)


def _mutate_nrows(dataframe, metadata, row_limits, prob):
    """ Mutate the number of rows an individual has by adding a new row and/or
    dropping a row at random so as not to exceed the bounds of
    :code:`row_limits`. """

    add = np.random.random()
    if add < prob and dataframe.shape[0] < row_limits[1]:
        dataframe, metadata = _add_line(dataframe, metadata, axis=0)

    remove = np.random.random()
    if remove < prob and dataframe.shape[0] > row_limits[0]:
        dataframe, metadata = _remove_line(dataframe, metadata, axis=0)

    return dataframe, metadata


def _mutate_ncols(dataframe, metadata, col_limits, pdfs, weights, prob):
    """ Mutate the number of columns an individual has by adding a new column
    and/or dropping a column at random. In either case, the bounds defined in
    :code:`col_limits` cannot be exceeded. """

    add = np.random.random()
    if isinstance(col_limits[1], tuple):
        condition = dataframe.shape[1] < sum(col_limits[1])
    else:
        condition = dataframe.shape[1] < col_limits[1]

    if add < prob and condition:
        dataframe, metadata = _add_line(
            dataframe, metadata, 1, col_limits, pdfs, weights
        )

    remove = np.random.random()
    if isinstance(col_limits[0], tuple):
        condition = dataframe.shape[1] > sum(col_limits[0])
    else:
        condition = dataframe.shape[1] > col_limits[0]

    if remove < prob and condition:
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
