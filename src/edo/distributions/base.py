""" .. Base inheritance class for all distributions. """

import abc


class Distribution(metaclass=abc.ABCMeta):
    """ An abstract base class for all currently implemented distributions and
    those defined by users. """

    @abc.abstractmethod
    def sample(self, nrows=None):
        """ A placeholder function for sampling from the distribution. """

    def to_dict(self):
        """ Returns a dictionary containing the name of distribution, and the
        values of all its parameters. """

        out = dict(vars(self))
        out["name"] = self.name

        return out
