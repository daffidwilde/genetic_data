""" Functions for the writing of generations and their fitnesses to file. """

from pathlib import Path

import dask
import pandas as pd


def _get_meta_df(metadata):
    """ Create a dataframe containing an individual's metadata. """

    max_params = max([len(vars(meta)) for meta in metadata])

    cols = ["family"]
    for i in range(max_params):
        cols.extend([f"param_{i}_name", f"param_{i}_value"])

    meta_df = pd.DataFrame([meta.to_tuple() for meta in metadata], columns=cols)
    return meta_df


def write_individual(individual, gen, idx, root):
    """ Write an individual to file. Each individual has their own directory at
    `root/gen/idx/` which contains their dataframe and metadata saved as in CSV
    files. """

    path = Path(f"{root}/{gen}/{idx}")
    path.mkdir(parents=True, exist_ok=True)

    dataframe, metadata = individual
    meta_df = _get_meta_df(metadata)

    dataframe.to_csv(path / "main.csv", index=False)
    meta_df.to_csv(path / "meta.csv", index=False)


def write_fitness(fitness, gen, root):
    """ Write the generation fitness to file in the generation's directory in
    `root`. """

    path = Path(f"{root}/{gen}")
    path.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(fitness, columns=[gen]).to_csv(
        path / "fitness.csv", index=False
    )


def write_generation(population, pop_fitness, gen, root, processes=None):
    """ Write all individuals in a generation and their collective fitnesses to
    file at the generation's directory in `root`. """

    tasks = (
        *[
            write_individual(individual, gen, i, root)
            for i, individual in enumerate(population)
        ], write_fitness(pop_fitness, gen, root)
    )

    if processes is None:
        dask.compute(*tasks, scheduler="single-threaded")
    else:
        dask.compute(*tasks, num_workers=processes)
