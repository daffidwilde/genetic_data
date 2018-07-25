import numpy as np

from genetic_data.components import create_individual
from genetic_data.operators import crossover, mutation
from genetic_data.pdfs import Normal

np.random.seed(1)

row_limits, col_limits = [1, 3], [1, 5]
pdfs = [Normal]

parents = [create_individual(row_limits, col_limits, pdfs) for _ in range(2)]
offspring = crossover(*parents, prob=0.5, pdfs=pdfs, weights=None)
mutant = mutation(
    offspring,
    prob=1.,
    row_limits=row_limits,
    col_limits=col_limits,
    pdfs=pdfs,
    weights=None,
    sigma=10.,
)

for i, parent in enumerate(parents):
    parent.round(4).to_csv(f"../reference/parent_{i+1}.csv")

offspring.round(4).to_csv("../reference/offspring.csv")
mutant.round(4).to_csv("../reference/mutant.csv")
