import numpy as np

from edo.individual import create_individual
from edo.operators import crossover, mutation
from edo.pdfs import Poisson

np.random.seed(0)

row_limits, col_limits = [1, 3], [1, 5]
pdfs = [Poisson]

parents = [create_individual(row_limits, col_limits, pdfs) for _ in range(2)]
offspring = crossover(*parents, prob=0.5)
mutant = mutation(
    offspring,
    prob=0.5,
    row_limits=row_limits,
    col_limits=col_limits,
    pdfs=pdfs,
    weights=None,
)

for i, parent in enumerate(parents):
    parent.dataframe.to_csv(f"../reference/parent_{i+1}.csv")

with open("../reference/parents.rst", "w") as parent_file:
    string = "And their metadata is::\n\n    "
    for parent in parents:
        string += "["
        for col in parent.metadata:
            string += col.__repr__() + ", "
        string = string[:-2]
        string += "] \n    "
    parent_file.write(string)

for ind, name in zip([offspring, mutant], ["offspring", "mutant"]):
    with open(f"../reference/{name}.rst", "w") as ind_file:
        string = "With the following metadata::\n\n    ["
        for col in ind.metadata:
            string += col.__repr__() + ", "
        string = string[:-2]
        string += "]"
        ind_file.write(string)

offspring.dataframe.to_csv("../reference/offspring.csv")
mutant.dataframe.to_csv("../reference/mutant.csv")
