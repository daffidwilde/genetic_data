""" Fitness-related functions. """


def get_fitness(fitness, population):
    """ Return the fitness score of each individual in a population. """

    return [fitness(individual.dataframe) for individual in population]
