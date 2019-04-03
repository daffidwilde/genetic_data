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
    param_limits : None
        A placeholder for a distribution parameter limit dictionary. These are
        considered the original limits and the class can be reset to them using
        the :code:`reset` class method.
    """

    name = "Distribution"
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

        setattr(Subtype, "__repr__", cls.__repr__)
        for key, value in vars(cls).items():
            if key != "subtypes":
                setattr(Subtype, key, copy.deepcopy(value))

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

        cls.subtypes = []

    def sample(self, nrows=None):
        """ Raise a :code:`NotImplementedError` by default. """

        raise NotImplementedError("You must define a sample method.")

    def to_tuple(self):
        """ Returns the name of distribution, and the names and values of all
        parameters as a tuple. This is used for the saving of data and little
        else. """

        out = [self.name]
        for key, val in vars(self).items():
            out.append(key)
            out.append(val)

        return tuple(out)
