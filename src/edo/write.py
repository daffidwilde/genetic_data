""" Functions for the writing of generations and their fitnesses to file. """

from pathlib import Path

import dask
import pandas as pd
import yaml


@dask.delayed
def write_individual(individual, gen, idx, root):
    """ Write an individual to file. Each individual has their own directory at
    `root/gen/idx/` which contains their dataframe and metadata saved as in CSV
    files. """

    path = Path(f"{root}/{gen}/{idx}")
    path.mkdir(parents=True, exist_ok=True)
    dataframe, metadata = individual

    dataframe.to_csv(path / "main.csv", index=False)
    with open(path / "main.meta", "w") as meta_file:
        yaml.dump([m.to_dict() for m in metadata], meta_file)


@dask.delayed
def write_fitness(fitness, gen, root):
    """ Write the generation fitness to file in the root directory. """

    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    size = len(fitness)

    if gen == 0:
        pd.DataFrame(
            {"fitness": fitness, "generation": gen, "individual": range(size)}
        ).to_csv(path / "fitness.csv", index=False)

    else:
        with open(path / "fitness.csv", "a") as fit_file:
            for fit, gen, ind in zip(fitness, [gen] * size, range(size)):
                fit_file.write(f"{fit},{gen},{ind}\n")


def write_generation(population, pop_fitness, gen, root, processes=None):
    """ Write all individuals in a generation and their collective fitnesses to
    file at the generation's directory in `root`. """

    tasks = (
        *[
            write_individual(individual, gen, i, root)
            for i, individual in enumerate(population)
        ],
        write_fitness(pop_fitness, gen, root),
    )

    if processes is None:
        dask.compute(*tasks, scheduler="single-threaded")
    else:
        dask.compute(*tasks, num_workers=processes)
