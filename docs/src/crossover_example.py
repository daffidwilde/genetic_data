""" .. Crossover example. """

import numpy as np

from edo.individual import create_individual
from edo.operators import crossover
from edo.pdfs import Poisson

np.random.seed(0)

row_limits, col_limits = [1, 3], [1, 5]
pdfs = [Poisson]

parents = [create_individual(row_limits, col_limits, pdfs) for _ in range(2)]

crossover_prob = 0.5

offspring = crossover(*parents, col_limits, pdfs, crossover_prob)

for i, parent in enumerate(parents):
    parent.dataframe.to_csv(f"../discussion/operators/parent_{i+1}.csv")

with open("../discussion/operators/parents.rst", "w") as parent_file:
    string = '.. :orphan:\n\n'
    string += "And their metadata is::\n\n    "
    for parent in parents:
        string += "["
        for col in parent.metadata:
            string += str(col) + ", "
        string = string[:-2]
        string += "]\n    "
    parent_file.write(string)

df, meta = offspring

df.to_csv('../discussion/operators/offspring.csv')

with open(f"../discussion/operators/offspring.rst", "w") as ind_file:
    string = '.. :orphan:\n\n'
    string += "With the following metadata::\n\n    ["
    for col in meta:
        string += str(col) + ", "
    string = string[:-2]
    string += "]"
    ind_file.write(string)
