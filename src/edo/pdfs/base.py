""" .. Base inheritance class for all distributions. """

from copy import deepcopy


class Distribution:
    """ A base class for all currently implemented distributions and those
    defined by users.

    Attributes
    ----------
    name : str
        The name of the distribution, "Distribution".
    param_limits : None
        A placeholder for a distribution parameter limit dictionary.
    """

    name = "Distribution"
    hard_limits = None
    param_limits = None

    def __repr__(self):

        params = ""
        for key, val in self.__dict__.items():
            params += f"{key}={val:.2f}, "
        params = params[:-2]

        return f"{self.name}({params})"

    @classmethod
    def reset(cls):
        """ Reset the class to have its original parameter limits, given in the
        hidden class attribute :code:`_param_limits`. """

        cls.param_limits = deepcopy(cls.hard_limits)

    def sample(self):
        """ Raise a :code:`NotImplementedError` by default. """

        raise NotImplementedError("You must define a sample method.")

    def to_tuple(self):
        """ Returns the name of distribution, and the names and values of all
        parameters as a tuple. This is used for the saving of data and little
        else. """

        out = [self.name]
        for key, val in self.__dict__.items():
            out.append(key)
            out.append(val)

        return tuple(out)
