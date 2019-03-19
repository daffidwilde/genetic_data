""" A function for the creation of pdf subtypes so that, e.g. multiple types of
the `Uniform` distribution may be present in the population. """

import copy


def build_class(cls):
    """ Build a new version of `cls` with identical properties that is
    independent of the original. """

    class Class:
        pass

    setattr(Class, "__repr__", cls.__repr__)
    setattr(Class, "family", cls)
    for key, value in vars(cls).items():
        if key != "subtypes":
            setattr(Class, key, copy.deepcopy(value))

    cls.subtypes.append(Class)
    return Class
