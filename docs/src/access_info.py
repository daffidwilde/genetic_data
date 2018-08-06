import numpy as np

from genetic_data.creation import create_individual
from genetic_data.pdfs import Normal, Poisson

np.random.seed(0)

individual = create_individual(row_limits=[3, 3], col_limits=[4, 4],
        pdfs=[Normal, Poisson])

df = individual.dataframe
meta = individual.column_metadata

df.round(4).to_csv('../how-to/access_dataframe.csv')
