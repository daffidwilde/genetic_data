""" Fitness-related functions. """

import dask
import edo


@dask.delayed
def get_fitness(dataframe, fitness, fitness_kwargs=None):
    """ Return the fitness score of the individual. """

    global cache
    cache = edo.cache

    if repr(dataframe) not in cache:
        if fitness_kwargs:
            cache[repr(dataframe)] = fitness(dataframe, **fitness_kwargs)
        else:
            cache[repr(dataframe)] = fitness(dataframe)

    return cache[repr(dataframe)]


def get_population_fitness(population, fitness, processes=None, fitness_kwargs=None):
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

    return out
