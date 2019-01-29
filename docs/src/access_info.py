""" .. How to access information from an individual. """

import numpy as np

from edo.individual import create_individual
from edo.pdfs import Normal, Poisson

np.random.seed(0)

individual = create_individual(
    row_limits=[3, 3], col_limits=[4, 4], pdfs=[Normal, Poisson]
)

df, meta = individual

with open("../how-to/access_dataframe.rst", "w") as ind_file:
    string = ".. :orphan:\n\n"
    string += "Then the dataframe can be accessed like this::\n\n    "
    string += ">>> individual.dataframe\n    "
    string += "".join([s + "\n    " for s in str(df).split("\n")])
    string += "\n    "
    ind_file.write(string)

with open("../how-to/access_metadata.rst", "w") as ind_file:
    string = ".. :orphan:\n\n"
    string += "And the metadata like this::\n\n    "
    string += ">>> individual.metadata\n    "
    string += str(meta)
    string += "\n    "
    ind_file.write(string)
