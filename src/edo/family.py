""" The distribution subtype handler. """

import os
import pathlib
import pickle

import numpy as np


class Family:
    """ A class for handling all concurrent subtypes of a distribution class. A
    subtype is an independent copy of the distribution class allowing more of
    the search space to be explored.

    Parameters
    ----------
    distribution : edo.distributions.Distribution
        The distribution class to keep track of. Must be of the same form as
        those in ``edo.distributions``.
    max_subtypes : int
        The maximum number of subtypes in the family that are currently being
        used in a run of the EA. There is no limit by default.

    Attributes
    ----------
    name : str
        The name of the family's distribution followed by ``Family``.
    subtype_id : int
        A counter that increments when new subtypes are created. Used as an
        identifier for a given subtype.
    subtypes : dict
        A dictionary that maps subtype identifiers to their corresponding
        subtype. This gets updated during a run to those that are currently
        being used in the population.
    all_subtypes : dict
        A dictionary of all subtypes that have been created in the family.
    random_state : np.random.RandomState
        The PRNG associated with this family to be used for the sampling and
        creation of subtypes.
    """

    def __init__(self, distribution, max_subtypes=None):

        self.distribution = distribution
        self.max_subtypes = max_subtypes

        self.name = distribution.name + "Family"
        self.subtype_id = 0
        self.subtypes = {}
        self.all_subtypes = {}
        self.random_state = np.random.mtrand._rand

    def __repr__(self):

        return f"{self.name}(subtypes={self.subtype_id})"

    def add_subtype(self, subtype_name=None, attributes=None):
        """ Create a copy of the distribution class that is identical and
        independent of the original. """

        if subtype_name is None:
            subtype_name = f"{self.distribution.name}Subtype"

        if attributes is None:
            attributes = _get_attrs_for_subtype(self.distribution)

        subtype = type(subtype_name, (self.distribution,), attributes)
        subtype.subtype_id = self.subtype_id
        subtype.family = self
        subtype.to_dict = _subtype_to_dict

        self.subtypes[self.subtype_id] = subtype
        self.all_subtypes[self.subtype_id] = subtype
        self.subtype_id += 1

    def make_instance(self, random_state):
        """ Select an existing subtype at random -- or create a new one if there
        is space available -- and return an instance of that subtype. """

        choices = list(self.subtypes)
        if self.max_subtypes is None or len(choices) < self.max_subtypes:
            choices.append(self.subtype_id)

        choice = self.random_state.choice(choices)
        if choice == self.subtype_id:
            self.add_subtype()

        instance = self.subtypes[choice](random_state)
        return instance

    def save(self, root=".edocache"):
        """ Save the current subtypes in the family and the family's random
        state in the ``root`` directory. """

        path = pathlib.Path(f"{root}/subtypes/{self.distribution.name}")
        path.mkdir(exist_ok=True, parents=True)

        with open(path / "state.pkl", "wb") as state:
            pickle.dump(
                self.random_state, state, protocol=pickle.HIGHEST_PROTOCOL
            )

        for subtype_id, subtype in self.all_subtypes.items():

            attributes = _get_attrs_for_subtype(subtype)
            with open(path / f"{subtype_id}.pkl", "wb") as sub:
                pickle.dump(attributes, sub, protocol=pickle.HIGHEST_PROTOCOL)

    def reset(self, root=None):
        """ Reset the family to have no subtypes and the default ``numpy`` PRNG.
        If ``root`` is passed then any cached information about the family is
        deleted. """

        self.subtype_id = 0
        self.subtypes.clear()
        self.all_subtypes.clear()
        self.random_state = np.random.mtrand._rand

        if root is not None:
            os.system(f"rm -r {root}/subtypes/{self.distribution.name}")

    @classmethod
    def load(cls, distribution, root=".edocache"):
        """ Load in any existing cached subtype dictionaries for
        ``distribution`` and restore the subtype along with the family's random
        state. """

        family = Family(distribution)
        name = distribution.name
        path = pathlib.Path(f"{root}/subtypes/{name}/")

        with open(path / "state.pkl", "rb") as state:
            family.random_state = pickle.load(state)

        subtype_paths = sorted(
            path.glob(r"[0-9]*.pkl"), key=lambda p: int(p.stem)
        )
        for path in subtype_paths:

            with open(path, "rb") as sub:
                attributes = pickle.load(sub)
                family.add_subtype(attributes=attributes)

        return family


def _get_attrs_for_subtype(obj):
    """ Get the attributes needed from ``obj`` to make or save a subtype. """

    attributes = {
        "name": obj.name,
        "dtype": obj.dtype,
        "hard_limits": obj.hard_limits,
        "param_limits": dict(obj.param_limits),
        "__init__": obj.__init__,
        "__repr__": obj.__repr__,
    }

    return attributes


def _subtype_to_dict(self):
    """ Convert an unpickleable subtype instance to a dictionary so it can be
    recovered at a later date. """

    attributes = {"name": self.name, "subtype_id": self.subtype_id}

    params = {}
    for param, val in vars(self).items():
        params[param] = val

    attributes["params"] = params

    return attributes
