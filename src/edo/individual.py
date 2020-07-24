""" A collection of objects to facilitate an individual representation. """

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .family import Family


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
    def from_file(cls, distributions, generation, index, root):
        """ Create an instance of `Individual` from files at `path`. """

        path = Path(root) / str(generation) / str(index)
        distributions = {dist.name: dist for dist in distributions}

        dataframe = pd.read_csv(path / "main.csv")

        with open(path / "main.meta", "r") as meta:
            meta_dicts = json.load(meta)

        root = path.parts[-3]
        metadata = []
        for meta in meta_dicts:
            distribution = meta["name"]
            family = globals().get(f"{distribution}Family", None)
            if family is None:
                distribution = distributions[distribution]
                family = Family.load(distribution, root)

            subtype_id = meta["subtype_id"]
            subtype = family.subtypes[subtype_id]

            pdf = subtype.__new__(subtype)
            pdf.__dict__.update(meta["params"])
            metadata.append(pdf)

        return Individual(dataframe, metadata)

    def to_file(self, generation, index, root):
        """ Write self to file. """

        path = Path(root) / str(generation) / str(index)
        path.mkdir(exist_ok=True, parents=True)

        self.dataframe.to_csv(path / "main.csv", index=False)

        meta_dicts = []
        for pdf in self.metadata:
            pdf.family.save(root)
            meta_dicts.append(pdf.to_dict())

        with open(path / "main.meta", "w") as meta:
            json.dump(meta_dicts, meta)

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
