""" Fitness-related functions. """


def get_fitness(fitness, population, fitness_kwargs=None):
    """ Return the fitness score of each individual in a population. """

    if fitness_kwargs:
        return [
            fitness(individual.dataframe, **fitness_kwargs)
            for individual in population
        ]

    return [fitness(individual.dataframe) for individual in population]
