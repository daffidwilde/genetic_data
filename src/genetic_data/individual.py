""" A collection of objects related to the definition and creation of an
individual in this GA. An individual is defined by a dataframe and its
associated metadata. This metadata is simply a list of the distributions from
which each column of the dataframe was generated. These are reused during
mutation and for filling in missing values during crossover. """

from collections import namedtuple

import numpy as np
import pandas as pd

Individual = namedtuple("Individual", ["dataframe", "metadata"])


def _sample_ncols(col_limits):
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


def _get_minimum_cols(cols, metadata, nrows, col_limits, pdfs, pdf_counts):
    """ If :code:`col_limits` has a tuple lower limit then sample columns of the
    correct class from :code:`pdfs` as needed to satisfy this bound. """

    for i, min_limit in enumerate(col_limits[0]):
        pdf_class = pdfs[i]
        for _ in range(min_limit):
            pdf = pdf_class()
            cols.append(pdf.sample(nrows))
            metadata.append(pdf)
            pdf_counts[pdf_class] += 1

    return cols, metadata, pdf_counts


def _get_remaining_cols(
    cols, metadata, nrows, ncols, col_limits, pdfs, weights, pdf_counts
):
    """ Sample all remaining columns for the current individual. If
    :code:`col_limits` has a tuple upper limit then sample all remaining
    columns for the individual without exceeding the bounds. """

    while len(cols) < ncols:
        pdf_class = np.random.choice(pdfs, p=weights)
        pdf = pdf_class()
        try:
            idx = pdfs.index(pdf_class)
            if pdf_counts[pdf_class] < col_limits[1][idx]:
                cols.append(pdf.sample(nrows))
                metadata.append(pdf)
                pdf_counts[pdf_class] += 1
        except TypeError:
            cols.append(pdf.sample(nrows))
            metadata.append(pdf)

    return cols, metadata


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
    ncols = _sample_ncols(col_limits)

    cols, metadata = [], []
    pdf_counts = {pdf_class: 0 for pdf_class in pdfs}

    if isinstance(col_limits[0], tuple):
        cols, metadata, pdf_counts = _get_minimum_cols(
            cols, metadata, nrows, col_limits, pdfs, pdf_counts
        )

    cols, metadata = _get_remaining_cols(
        cols, metadata, nrows, ncols, col_limits, pdfs, weights, pdf_counts
    )

    dataframe = pd.DataFrame({i: col for i, col in enumerate(cols)})
    return Individual(dataframe, metadata)
