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

    def __init__(self, dataframe, metadata, random_state=None):

        self.dataframe = dataframe
        self.metadata = metadata

        if random_state is None:
            random_state = np.random.mtrand._rand

        self.random_state = random_state

    def __repr__(self):

        return (
            f"Individual(dataframe={self.dataframe}, metadata={self.metadata})"
        )

    def __iter__(self):

        for part in [self.dataframe, self.metadata]:
            yield part

    @classmethod
    def from_file(cls, path, distributions, cache_dir=".edocache", method=pd):
        """ Create an instance of `Individual` from files at `path`. """

        path = Path(path)
        distributions = {dist.name: dist for dist in distributions}

        dataframe = method.read_csv(path / "main.csv")
        dataframe.columns = map(int, dataframe.columns)

        with open(path / "main.meta", "r") as meta:
            meta_dicts = json.load(meta)

        metadata = []
        for meta in meta_dicts:
            distribution = meta["name"]
            family = globals().get(f"{distribution}Family", None)
            if family is None:
                distribution = distributions[distribution]
                family = Family.load(distribution, cache_dir)

            subtype_id = meta["subtype_id"]
            subtype = family.subtypes[subtype_id]

            pdf = subtype.__new__(subtype)
            pdf.__dict__.update(meta["params"])
            metadata.append(pdf)

        with open(path / "main.state", "rb") as state:
            random_state = pickle.load(state)

        return Individual(dataframe, metadata, random_state)

    def to_file(self, path, cache_dir=".edocache"):
        """ Write self to file. """

        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)

        self.dataframe.to_csv(path / "main.csv", index=False)

        meta_dicts = []
        for pdf in self.metadata:
            pdf.family.save(cache_dir)
            meta_dicts.append(pdf.to_dict())

        with open(path / "main.meta", "w") as meta:
            json.dump(meta_dicts, meta)

        with open(path / "main.state", "wb") as state:
            pickle.dump(
                self.random_state, state, protocol=pickle.HIGHEST_PROTOCOL
            )

        return path


def _sample_ncols(col_limits, random_state):
    """ Sample a valid number of columns from the column limits, even if those
    limits contain tuples. """

    integer_limits = []
    for lim in col_limits:
        try:
            integer_lim = sum(lim)
        except TypeError:
            integer_lim = lim
        integer_limits.append(integer_lim)

    return random_state.randint(integer_limits[0], integer_limits[1] + 1)


def _get_minimum_columns(
    nrows, col_limits, families, family_counts, random_state
):
    """ If :code:`col_limits` has a tuple lower limit then sample columns of the
    correct class from :code:`families` as needed to satisfy this bound. """

    columns, metadata = [], []
    for family, min_limit in zip(families, col_limits[0]):
        for _ in range(min_limit):
            meta = family.make_instance(random_state)
            columns.append(meta.sample(nrows, random_state))
            metadata.append(meta)
            family_counts[family.name] += 1

    return columns, metadata, family_counts


def _get_remaining_columns(
    columns,
    metadata,
    nrows,
    ncols,
    col_limits,
    families,
    weights,
    family_counts,
    random_state,
):
    """ Sample all remaining columns for the current individual. If
    :code:`col_limits` has a tuple upper limit then sample all remaining
    columns for the individual without exceeding the bounds. """

    while len(columns) < ncols:
        family = random_state.choice(families, p=weights)
        idx = families.index(family)
        try:
            if family_counts[family.name] < col_limits[1][idx]:
                meta = family.make_instance(random_state)
                columns.append(meta.sample(nrows, random_state))
                metadata.append(meta)
                family_counts[family.name] += 1

        except TypeError:
            meta = family.make_instance(random_state)
            columns.append(meta.sample(nrows, random_state))
            metadata.append(meta)

    return columns, metadata


def create_individual(row_limits, col_limits, families, weights, random_state):
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
    random_state : numpy.random.RandomState
        The PRNG associated with the individual to use for random sampling.
    """

    nrows = random_state.randint(row_limits[0], row_limits[1] + 1)
    ncols = _sample_ncols(col_limits, random_state)

    columns, metadata = [], []
    family_counts = {family.name: 0 for family in families}

    if isinstance(col_limits[0], tuple):
        columns, metadata, pdf_counts = _get_minimum_columns(
            nrows, col_limits, families, family_counts, random_state
        )

    columns, metadata = _get_remaining_columns(
        columns,
        metadata,
        nrows,
        ncols,
        col_limits,
        families,
        weights,
        family_counts,
        random_state,
    )

    dataframe = pd.DataFrame({i: col for i, col in enumerate(columns)})
    return Individual(dataframe, metadata, random_state)
