""" Fitness-related functions. """

from pathlib import Path

import dask
import pandas as pd

import edo


@dask.delayed
def get_fitness(dataframe, fitness, fitness_kwargs=None):
    """ Return the fitness score of the individual. """

    cache = edo.cache
    key = repr(dataframe)

    if key not in cache:
        if fitness_kwargs:
            cache[key] = fitness(dataframe, **fitness_kwargs)
        else:
            cache[key] = fitness(dataframe)

    return cache[key]


def get_population_fitness(
    population, fitness, processes=None, fitness_kwargs=None
):
    """ Return the fitness of each individual in the population. This can be
    done in parallel by specifying a number of cores to use for independent
    processes. """

    tasks = (
        get_fitness(individual.dataframe, fitness, fitness_kwargs)
        for individual in population
    )

    if processes is None:
        out = dask.compute(*tasks, scheduler="single-threaded")
    else:
        out = dask.compute(*tasks, num_workers=processes)

    return list(out)


@dask.delayed
def write_fitness(fitness, generation, root):
    """ Write the generation fitness to file in the root directory. """

    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    size = len(fitness)

    if generation == 0:
        pd.DataFrame(
            {
                "fitness": fitness,
                "generation": generation,
                "individual": range(size),
            }
        ).to_csv(path / "fitness.csv", index=False)

    else:
        with open(path / "fitness.csv", "a") as fit_file:
            for fit, gen, ind in zip(fitness, [generation] * size, range(size)):
                fit_file.write(f"{fit},{gen},{ind}\n")
