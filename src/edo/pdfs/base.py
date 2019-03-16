""" .. Base inheritance class for all distributions. """


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
    def reset(cls):
        """ Reset the class to have its original parameter limits, i.e. those
        given in the class attribute :code:`param_limits` when the first
        instance is made. """

        cls.subtypes = []

    def sample(self, nrows=None):
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
