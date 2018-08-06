""" Default crossover and mutation operators for a genetic algorithm. """

from copy import deepcopy

import numpy as np
import pandas as pd

from genetic_data.individual import Individual


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


def _remove_line(dataframe, axis, metadata):
    """ Remove a line (row or column) from a dataset at random. """

    if axis == 0:
        line = np.random.choice(dataframe.index)
    else:
        line = np.random.choice(dataframe.columns)
        idx = dataframe.columns.get_loc(line)
        metadata.pop(idx)

    dataframe = _rename(dataframe.drop(line, axis=axis), axis)
    return dataframe, metadata


def _add_line(dataframe, axis, metadata=None, pdfs=None, weights=None):
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
    else:
        pdf = np.random.choice(pdfs, p=weights)()
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


def crossover(parent1, parent2, prob):
    """ Select information from `parent1` with probability `prob`. Otherwise
    select from `parent2`. Collate information to form a new individual. """

    metadata1, dataframe1 = parent1
    metadata2, dataframe2 = parent2

    widest = np.argmax([len(dataframe1.columns), len(dataframe2.columns)])
    widest_metadata, widest_dataframe = parent2 if widest else parent1

    if np.random.random() < prob:
        nrows = len(dataframe1)
    else:
        nrows = len(dataframe2)

    if np.random.random() < prob:
        ncols = len(dataframe1.columns)
    else:
        ncols = len(dataframe2.columns)

    metadata, dataframe = [], pd.DataFrame()

    for i in range(ncols):
        if i < min([len(dataframe1.columns), len(dataframe2.columns)]):
            if np.random.random() < prob:
                dataframe_col = dataframe1[f"col_{i}"]
                col_pdf = metadata1[i]
            else:
                dataframe_col = dataframe2[f"col_{i}"]
                col_pdf = metadata2[i]
        else:
            dataframe_col = widest_dataframe[f"col_{i}"]
            col_pdf = widest_metadata[i]

        dataframe[f"col_{i}"] = dataframe_col
        metadata.append(col_pdf)

    while len(dataframe) != nrows:
        if len(dataframe) > nrows:
            dataframe = dataframe.iloc[:nrows, :]
        elif len(dataframe) < nrows:
            dataframe, metadata = _add_line(dataframe, 0, metadata)

    dataframe = _fillna(dataframe, metadata)
    return Individual(metadata, dataframe)


def mutation(individual, prob, row_limits, col_limits, pdfs, weights):
    """ Mutate an individual. Here, the characteristics of an individual can be
    split into two parts: their dimensions, and their values. Each of these
    parts is mutated in a different way using the same probability, `prob`. """

    metadata, dataframe = deepcopy(individual)

    limits = [row_limits, col_limits]
    for axis in [0, 1]:

        # Try to remove a line at random
        r_remove = np.random.random()
        if r_remove < prob and dataframe.shape[axis] > limits[axis][0]:
            dataframe, metadata = _remove_line(dataframe, axis, metadata)

        # Try to add a line to the end of axis
        r_add = np.random.random()
        if r_add < prob and dataframe.shape[axis] < limits[axis][1]:
            dataframe, metadata = _add_line(
                dataframe, axis, metadata, pdfs, weights
            )

    # Iterate over the elements of the dataframe, mutating them by resampling
    # from each column's associated distribution in `metadata`.
    for j, col in enumerate(dataframe.columns):
        pdf = metadata[j]
        for i, value in enumerate(dataframe[col]):
            if np.random.random() < prob:
                value = pdf.sample(1)[0]
                dataframe.iloc[i, j] = value

    return Individual(metadata, dataframe)
