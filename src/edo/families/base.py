""" .. Base inheritance class for all distributions. """

import copy

import numpy as np


class Distribution:
    """ A base class for all currently implemented distributions and those
    defined by users.

    Attributes
    ----------
    name : str
        The name of the distribution, :code:`"Distribution"`.
    subtype_id : int
        A placeholder for a subtype's identifier. Defaults to 0 and increments
        with each new subtype. Reset with :code:`reset` class method.
    subtypes : list
        A list of all currently available subtypes. Can be reset with
        :code:`reset` class method.
    max_subtypes : int
        The maximum number of subtypes the distribution may have at any given
        time. If :code:`None`, then no limit is set.
    param_limits : None
        A placeholder for a distribution parameter limit dictionary. These are
        considered the original limits to be used by all subtypes of the
        distribution.
    """

    name = "Distribution"
    subtype_id = 0
    subtypes = []
    max_subtypes = None
    param_limits = None

    def __repr__(self):

        params = ""
        for name, value in vars(self).items():
            if isinstance(value, list):
                params += f"{name}=["
                for val in value:
                    params += f"{val:.2f}, "
                params = params[:-2]
                params += "], "
            else:
                params += f"{name}={value:.2f}, "

        params = params[:-2]
        return f"{self.name}({params})"

    @classmethod
    def build_subtype(cls):
        """ Build a copy of the distribution class with identical properties
        that is independent of the original. """

        class Subtype:

            family = cls
            subtype_id = cls.subtype_id

        setattr(Subtype, "__repr__", cls.__repr__)
        setattr(Subtype, "sample", cls.sample)
        setattr(Subtype, "to_dict", cls.to_dict)
        for key, value in vars(cls).items():
            if "subtype" not in key:
                setattr(Subtype, key, copy.deepcopy(value))

        cls.subtype_id += 1
        cls.subtypes.append(Subtype)
        return Subtype

    @classmethod
    def make_instance(cls):
        """ Choose an existing subtype or build a new one if there is space
        available in their subtype list. Return an instance of that subtype. """

        choices = list(cls.subtypes)
        if cls.max_subtypes is None or len(choices) < cls.max_subtypes:
            choices.append(cls.build_subtype)

        subtype = np.random.choice(choices)
        if subtype == cls.build_subtype:
            subtype = cls.build_subtype()

        return subtype()

    @classmethod
    def reset(cls):
        """ Reset the class to have no subtypes. """

        cls.subtype_id = 0
        cls.subtypes = []

    def sample(self, nrows=None):
        """ Raise a :code:`NotImplementedError` by default. """

        raise NotImplementedError("You must define a sample method.")

    def to_dict(self):
        """ Returns a dictionary containing the name of distribution, and the
        values of all its parameters. """

        out = dict(vars(self))
        out["name"] = self.name
        out["subtype_id"] = self.subtype_id

        return out
