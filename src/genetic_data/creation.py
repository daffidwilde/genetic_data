""" A script containing functions for each of the components of the genetic
algorithm. """

import numpy as np
import pandas as pd

from genetic_data.individual import Individual
from genetic_data.operators import crossover, mutation


def _get_ncols(col_limits):
    """ Sample a valid number of columns from the column limits, even if those
    limits contain tuples. """

    integer_limits = []
    for lim in col_limits:
        try:
            integer_lim = sum(lim)
        except TypeError:
            integer_lim = lim
        integer_limits.append(integer_lim)

    return np.random.randint(integer_limits[0], integer_limits[1] + 1)


def create_individual(row_limits, col_limits, pdfs, weights=None):
    """ Create an individual dataset-metadata representation within the limits
    provided. An individual is contained within a :code:`namedtuple` object.

    Parameters
    ----------
    row_limits : list
        Lower and upper bounds on the number of rows a dataset can have.
    col_limits : list
        Lower and upper bounds on the number of columns a dataset can have.
        Tuples can be used to indicate limits on the number of columns needed to 
    pdfs : list
        A list of potential column pdf classes to select from such as those
        found in :code:`genetic_data.pdfs`.
    weights : list
        A sequence of relative weights the same length as :code:`pdfs`. This
        acts as a probability distribution from which to sample column classes.
        If :code:`None`, column classes are sampled uniformly.
    """

    nrows = np.random.randint(row_limits[0], row_limits[1] + 1)
    ncols = _get_ncols(col_limits)

    curr_col = 0
    metadata, dataframe = [], pd.DataFrame()
    pdf_counts = {pdf_class: 0 for pdf_class in pdfs}

    if isinstance(col_limits[0], tuple):
        for i, min_limit in enumerate(col_limits[0]):
            for _ in range(min_limit):
                pdf_class = pdfs[i]
                pdf = pdf_class()
                dataframe[f"col_{curr_col}"] = pdf.sample(nrows)
                metadata.append(pdf)
                pdf_counts[pdf_class] += 1
                curr_col += 1

    if isinstance(col_limits[1], tuple):
        while curr_col < ncols:
            pdf_class = np.random.choice(pdfs, p=weights, size=1)[0]
            idx = pdfs.index(pdf_class)
            pdf = pdf_class()
            if pdf_counts[pdf_class] < col_limits[1][idx]:
                dataframe[f"col_{curr_col}"] = pdf.sample(nrows)
                metadata.append(pdf)
                pdf_counts[pdf_class] += 1
                curr_col += 1
    else:
        for i, pdf_class in enumerate(
            np.random.choice(pdfs, p=weights, size=ncols - curr_col)
        ):
            i += curr_col
            pdf = pdf_class()
            dataframe[f"col_{i}"] = pdf.sample(nrows)
            metadata.append(pdf)

    return Individual(metadata, dataframe)


def create_initial_population(size, row_limits, col_limits, pdfs, weights=None):
    """ Create an initial population for the genetic algorithm based on the
    given parameters.

    Parameters
    ----------
    size : int
        The number of individuals in the population.
    row_limits : list
        Limits on the number of rows a dataset can have.
    col_limits : list
        Limits on the number of columns a dataset can have.
    pdfs : list
        A list of potential column pdf classes such as those found in
        `pdfs.py`. Must have a `.sample()` and `.mutate()` method.
    weights : list
        A sequence of relative weights the same length as `column_classes`. This
        acts as a loose probability distribution from which to sample column
        classes. If `None`, column classes are sampled equally.

    Returns
    -------
    population : list
        A collection of individuals.
    """

    if size <= 1:
        raise ValueError(
            "There must be more than one individual in a \
                          population"
        )

    population = [
        create_individual(row_limits, col_limits, pdfs, weights)
        for _ in range(size)
    ]

    return population


def create_new_population(
    parents,
    size,
    crossover_prob,
    mutation_prob,
    row_limits,
    col_limits,
    pdfs,
    weights,
):
    """ Given a set of potential parents to be carried into the next generation,
    create offspring from pairs within that set until there are enough
    individuals. Each individual offspring is formed using a crossover operator
    on the two parent individuals and then mutating them according to the
    probability `mutation_prob`. """

    population = parents
    while len(population) < size:
        parent1_idx, parent2_idx = np.random.choice(len(parents), size=2)
        parent1, parent2 = parents[parent1_idx], parents[parent2_idx]
        offspring = crossover(parent1, parent2, crossover_prob)
        mutant = mutation(
            offspring, mutation_prob, row_limits, col_limits, pdfs, weights
        )
        population.append(mutant)

    return population
