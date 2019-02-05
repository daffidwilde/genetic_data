""" .. Mutation example. """

import numpy as np

from edo.individual import create_individual
from edo.operators import mutation
from edo.pdfs import Poisson

np.random.seed(0)

row_limits, col_limits = [1, 3], [1, 5]
pdfs = [Poisson]

individual = create_individual(row_limits, col_limits, pdfs)

mutation_prob = 0.5

mutant = mutation(individual, mutation_prob, row_limits, col_limits, pdfs)

for name, ind in zip(["individual", "mutant"], [individual, mutant]):
    df, meta = ind

    df.to_csv(f"../discussion/operators/{name}.csv")

    with open(f"../discussion/operators/{name}.rst", "w") as ind_file:
        string = ".. :orphan:\n\n"
        string += "And their metadata is::\n\n    ["
        for col in meta:
            string += str(col) + ", "
        string = string[:-2]
        string += "]\n    "
        ind_file.write(string)
