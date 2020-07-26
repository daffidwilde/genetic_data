""" The base class from which all distributions inherit. """

import abc


class Distribution(metaclass=abc.ABCMeta):
    """ An abstract base class for all currently implemented distributions and
    those defined by users. """

    @abc.abstractmethod
    def sample(self, nrows=None, random_state=None):
        """ A placeholder function for sampling from the distribution. """
