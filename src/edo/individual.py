""" A collection of objects related to the definition and creation of an
individual in this EA. An individual is defined by a dataframe and its
associated metadata. This metadata is simply a list of the distributions from
which each column of the dataframe was generated. These are reused during
mutation and for filling in missing values during crossover. """

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


class Individual:
    """ A class to represent an individual in the EA.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe of the individual.
    metadata : list
        A list of distributions that are associated with the respective column
        of `dataframe`.
    """

    def __init__(self, dataframe, metadata):

        self.dataframe = dataframe
        self.metadata = metadata

    def __repr__(self):

        return (
            f"Individual(dataframe={self.dataframe}, metadata={self.metadata})"
        )

    def __iter__(self):

        for _, val in vars(self).items():
            yield val

    @classmethod
    def from_file(cls, path):
        """ Create an instance of `Individual` from files at `path`. """

        dataframe = pd.read_csv(path / "main.csv")
        with open(path / "main.meta", "r") as meta_file:
            metadata = yaml.load(meta_file, Loader=yaml.FullLoader)

        return Individual(dataframe, metadata)

    def to_history(self):
        """ Export a copy of itself fit for a population history, i.e. with
        dictionary metadata as sampling is no longer required. """

        meta_dicts = [pdf.to_dict() for pdf in self.metadata]
        return Individual(self.dataframe, meta_dicts)

    def to_file(self, generation, index, root):
        """ Write self to file. """

        path = Path(root) / str(generation) / str(index)
        path.mkdir(exist_ok=True, parents=True)

        dataframe, metadata = self.to_history()
        dataframe.to_csv(path / "main.csv", index=False)
        with open(path / "main.meta", "w") as meta_file:
            yaml.dump(metadata, meta_file)

        return path


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


def _get_minimum_columns(nrows, col_limits, families, family_counts):
    """ If :code:`col_limits` has a tuple lower limit then sample columns of the
    correct class from :code:`families` as needed to satisfy this bound. """

    columns, metadata = [], []
    for family, min_limit in zip(families, col_limits[0]):
        for _ in range(min_limit):
            meta = family.make_instance()
            columns.append(meta.sample(nrows))
            metadata.append(meta)
            family_counts[family.name] += 1

    return columns, metadata, family_counts


def _get_remaining_columns(
    columns, metadata, nrows, ncols, col_limits, families, weights, family_counts
):
    """ Sample all remaining columns for the current individual. If
    :code:`col_limits` has a tuple upper limit then sample all remaining
    columns for the individual without exceeding the bounds. """

    while len(columns) < ncols:
        family = np.random.choice(families, p=weights)
        idx = families.index(family)
        try:
            if family_counts[family.name] < col_limits[1][idx]:
                meta = family.make_instance()
                columns.append(meta.sample(nrows))
                metadata.append(meta)
                family_counts[family.name] += 1

        except TypeError:
            meta = family.make_instance()
            columns.append(meta.sample(nrows))
            metadata.append(meta)

    return columns, metadata


def create_individual(row_limits, col_limits, families, weights=None):
    """ Create an individual dataset-metadata representation within the limits
    provided. An individual is contained within a :code:`namedtuple` object.

    Parameters
    ----------
    row_limits : list
        Lower and upper bounds on the number of rows a dataset can have.
    col_limits : list
        Lower and upper bounds on the number of columns a dataset can have.
        Tuples can be used to indicate limits on the number of columns needed to
    families : list
        A list of `edo.Family` instances handling the column distributions that
        can be selected from.
    weights : list
        A sequence of relative weights the same length as :code:`families`. This
        acts as a probability distribution from which to sample column classes.
        If :code:`None`, column classes are sampled uniformly.
    """

    nrows = np.random.randint(row_limits[0], row_limits[1] + 1)
    ncols = _sample_ncols(col_limits)

    cols, metadata = [], []
    family_counts = {family.name: 0 for family in families}

    if isinstance(col_limits[0], tuple):
        cols, metadata, pdf_counts = _get_minimum_columns(
            nrows, col_limits, families, family_counts
        )

    cols, metadata = _get_remaining_columns(
        cols, metadata, nrows, ncols, col_limits, families, weights, family_counts
    )

    dataframe = pd.DataFrame({i: col for i, col in enumerate(cols)})
    return Individual(dataframe, metadata)
