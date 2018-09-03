""" .. Base inheritance class for all distributions. """

class Distribution():
    """ A base class for all currently implemented distributions and those
    defined by users.

    Attributes
    ----------
    name : str
        The name of the distribution, "Distribution".
    """

    name = "Distribution"

    def __repr__(self):

        params = ""
        for key, val in self.__dict__.items():
            params += f"{key}={val:.2f}, "
        params = params[:-2]

        return f"{self.name}({params})"

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
