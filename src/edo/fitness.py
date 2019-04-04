""" Fitness-related functions. """


def get_fitness(dataframe, fitness, cache, fitness_kwargs=None):
    """ Return the fitness score of the individual. """

    if repr(dataframe) not in cache:
        if fitness_kwargs:
            cache[repr(dataframe)] = fitness(dataframe, **fitness_kwargs)
        else:
            cache[repr(dataframe)] = fitness(dataframe)

    return cache[repr(dataframe)]


def get_population_fitness(population, fitness, cache, fitness_kwargs=None):
    """ Return the fitness of each individual in the population. """

    return [
        get_fitness(individual.dataframe, fitness, cache, fitness_kwargs)
        for individual in population
    ]
