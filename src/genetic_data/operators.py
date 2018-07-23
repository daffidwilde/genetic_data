""" Default crossover and mutation operators for a genetic algorithm. """

import numpy as np
import pandas as pd


def _rename(df, axis):
    """ Rename columns or reindex to make sense after deletion or addition of a
    new line. """

    if axis == 0:
        df.reset_index(drop=True)
    else:
        df.columns = [f"col_{i}" for i in range(len(df.columns))]

    return df


def _fillna_random(df, pdfs=None, weights=None):
    """ Fill in `NaN` values of a column by sampling from the real values in
    that column. If no such values are available, then create a new column. """

    for col in df.columns:
        data = df[col]
        if data.isnull().any():
            nulls = data.isnull()
            if data[-nulls].size > 0:
                samples = np.random.choice(
                    data[-nulls].values, size=nulls.sum()
                )
                data[nulls] = samples
            else:
                pdf = np.random.choice(pdfs, p=weights)
                data[col] = pdf().sample(len(df))

    return df


def _remove_line(df, axis):
    """ Remove a line (row or column) from a dataset at random. """

    if axis == 0:
        line = np.random.choice(df.index)
    else:
        line = np.random.choice(df.columns)

    df = _rename(df.drop(line, axis=axis), axis)
    return df


def _add_line(df, axis, pdfs=None, weights=None):
    """ Add a line (row or column) to the end of a dataset. Rows are added by
    adding a row of `NaN`s and then filling these with values in the respective
    columns where possible. Columns are added in the same way that they are at
    the initial creation of an individual. """

    nrows, ncols = df.shape

    if axis == 0:
        df = df.append(
            {f"col_{i}": np.nan for i in range(ncols)}, ignore_index=True
        )
    else:
        pdf = np.random.choice(pdfs, p=weights)
        df[f"col_{ncols + 1}"] = pdf().sample(nrows)

    df = _fillna_random(_rename(df, axis), pdfs, weights)
    return df


def crossover(parent1, parent2, prob, pdfs, weights):
    """ Select "alleles" from `parent1` with probability `prob`. Otherwise
    select from `parent2`. Collate alleles to form a new individual. """

    widest_parent = sorted(
        [parent1, parent2], key=lambda x: len(x.columns), reverse=True
    )[0]

    if np.random.random() < prob:
        nrows = len(parent1)
    else:
        nrows = len(parent2)

    if np.random.random() < prob:
        ncols = len(parent1.columns)
    else:
        ncols = len(parent2.columns)

    individual = pd.DataFrame({f"col_{i}": [] for i in range(ncols)})
    for i in range(ncols):
        if i < min([len(parent1.columns), len(parent2.columns)]):
            if np.random.random() < prob:
                individual[f"col_{i}"] = parent1[f"col_{i}"]
            else:
                individual[f"col_{i}"] = parent2[f"col_{i}"]
        else:
            individual[f"col_{i}"] = widest_parent[f"col_{i}"]

    while len(individual) != nrows:
        if len(individual) > nrows:
            individual = _remove_line(individual, axis=0)
        elif len(individual) < nrows:
            individual = _add_line(individual, axis=0)

    individual = _fillna_random(individual, pdfs, weights)
    return individual


def mutation(individual, prob, row_limits, col_limits, pdfs, weights, sigma=1):
    """ Mutate an individual. Here, the characteristics of an individual can be
    split into two parts: their dimensions, and their values. Each of these
    parts is mutated in a different way using the same probability, `prob`. """

    # Add or remove a row or column at random.
    limits = [row_limits, col_limits]
    for axis in [0, 1]:
        r = np.random.random()
        if r < prob / 2 and len(individual) > limits[axis][0]:
            individual = _remove_line(individual, axis=0)
        elif r < prob / 2 and len(individual) < limits[axis][1]:
            individual = _add_line(individual, axis, pdfs, weights)

    # Iterate over the elements of an individual, mutating them by resampling
    # from the normal distribution with standard deviation `sigma` around the
    # current value.
    for col in individual.columns:
        for value in individual[col]:
            if np.random.random() < prob:
                value = np.random.normal(loc=value, scale=sigma)

    return individual
