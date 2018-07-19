""" Tests for the components of the algorithm. """

import pandas as pd
import pytest

from hypothesis import given, settings
from hypothesis.strategies import integers
from genetic_data.pdfs import Gamma, Poisson
from genetic_data.components import create_individual, \
                                    create_initial_population, \
                                    get_dataframes, \
                                    get_fitness, \
                                    get_ordered_population, \
                                    select_parents, \
                                    create_offspring, \
                                    mutate_population

from test_util.trivials import trivial_fitness
from test_util.parameters import individual_limits, \
                                 population_limits, \
                                 ind_fitness_limits, \
                                 pop_fitness_limits, \
                                 selection_limits, \
                                 small_props_limits, \
                                 offspring_limits, \
                                 mutation_limits


class TestCreation():
    """ Tests for the creation of an individual and an initial population. """

    @individual_limits
    def test_individual(self, row_limits, col_limits, weights):
        """ Create an individual and verify that it is a list of the correct
        length with the right characteristics. """

        pdfs = [Gamma, Poisson]
        individual = create_individual(row_limits, col_limits, pdfs, weights)
        assert isinstance(individual, tuple)
        assert len(individual) == individual[1] + 2
        assert isinstance(individual[0], int) and isinstance(individual[1], int)

        for col in individual[2:]:
            assert isinstance(col, tuple(pdfs))

    @population_limits
    def test_initial_population(self, size, row_limits, col_limits, weights):
        """ Create an initial population of individuals and verify it is a list
        of the correct length with the right characteristics. """

        pdfs = [Gamma, Poisson]
        population = create_initial_population(size, row_limits, col_limits,
                                               pdfs, weights)
        assert isinstance(population, list)
        assert len(population) == size

        for ind in population:
            assert len(ind) == ind[1] + 2
            assert isinstance(ind[0], int) and isinstance(ind[1], int)

            for col in ind[2:]:
                assert isinstance(col, tuple(pdfs))

    @given(size=integers(max_value=1))
    def test_too_small_population(self, size):
        """ Verify that a `ValueError` is raised for small population sizes. """

        with pytest.raises(ValueError):
            population = create_initial_population(size, None, None, None, None)


class TestGetFitness():
    """ Test the get_fitness function. """

    @ind_fitness_limits
    def test_get_dataframes(self, row_limits, col_limits, weights, max_seed):
        """ Verify that an individual's expanded form is a `pandas.DataFrame`
        object of the correct shape. """

        pdfs = [Gamma, Poisson]
        individual = create_individual(row_limits, col_limits, pdfs, weights)
        dfs = get_dataframes(individual, max_seed)
        assert len(dfs) == max_seed

        for df in dfs:
            assert isinstance(df, pd.DataFrame)
            assert df.shape == (individual[0], individual[1])

    @pop_fitness_limits
    def test_get_fitness(self, size, row_limits, col_limits, weights, max_seed):
        """ Create a population and get its fitness. Then verify that the
        fitness is of the correct size and data type. """

        pdfs = [Gamma, Poisson]
        population = create_initial_population(size, row_limits, col_limits,
                                               pdfs, weights)
        population_fitness = get_fitness(trivial_fitness, population, max_seed)
        assert population_fitness.shape == (size,)
        assert population_fitness.dtype == 'float'

    @pop_fitness_limits
    def test_get_ordered_population(self, size, row_limits, col_limits,
                                    weights, max_seed):
        """ Create a population, get its fitness and order the individuals in
        descending order of their fitness. Verify that all individuals are
        there. """

        pdfs = [Gamma, Poisson]
        population = create_initial_population(size, row_limits, col_limits,
                                               pdfs, weights)
        population_fitness = get_fitness(trivial_fitness, population, max_seed)
        ordered_population = get_ordered_population(population,
                                                    population_fitness)
        assert set(ordered_population.keys()) == set(population)


class TestBreedingProcess():
    """ Test the parent selection and offspring creation processes, including
    the mutation of the new offspring population. """

    @selection_limits
    def test_select_parents(self, size, row_limits, col_limits, weights,
                            props, max_seed):
        """ Create a population, get its fitness and select potential parents
        based on that fitness vector. Verify that parents are selected without
        replacement. """

        best_prop, lucky_prop = props
        pdfs = [Gamma, Poisson]
        population = create_initial_population(size, row_limits, col_limits,
                                               pdfs, weights)
        population_fitness = get_fitness(trivial_fitness, population, max_seed)
        parents = select_parents(population, population_fitness,
                                 best_prop, lucky_prop)

        ind_counts = {ind: 0 for ind in population}
        while parents != []:
            for ind in population:
                if ind in parents:
                    ind_counts[ind] += 1
                    parents.remove(ind)
        for ind in ind_counts:
            assert ind_counts[ind] in [0, 1]

    @small_props_limits
    def test_select_parents_raises_error(self, size, row_limits, col_limits,
                                         weights, props, max_seed):
        """ Assert that best and lucky proportions must be sensible. """

        with pytest.raises(ValueError):
            best_prop, lucky_prop = props
            pdfs = [Gamma, Poisson]
            population = create_initial_population(size, row_limits, col_limits,
                                                   pdfs, weights)
            population_fitness = get_fitness(trivial_fitness, population,
                                             max_seed)

            parents = select_parents(population, population_fitness,
                                     best_prop, lucky_prop)

    @offspring_limits
    def test_create_offspring(self, size, row_limits, col_limits, weights,
                              props, prob, max_seed):
        """ Create a population and use them to create a new proto-population
        of offspring. Verify that each offspring is an individual and their are
        the correct number of them. That way, this collection of offspring are
        in fact a population. """

        best_prop, lucky_prop = props
        pdfs = [Gamma, Poisson]
        population = create_initial_population(size, row_limits, col_limits,
                                               pdfs, weights)
        population_fitness = get_fitness(trivial_fitness, population, max_seed)
        parents = select_parents(population, population_fitness,
                                 best_prop, lucky_prop)
        offspring = create_offspring(parents, prob, size)
        assert isinstance(offspring, list)
        assert len(offspring) == size

        for ind in offspring:
            assert len(ind) == ind[1] + 2
            assert isinstance(ind[0], int) and isinstance(ind[1], int)

            for col in ind[2:]:
                assert isinstance(col, tuple(pdfs))

    @mutation_limits
    @settings(deadline=None)
    def test_mutate_population(self, size, row_limits, col_limits, weights,
                               mutation_prob, allele_prob):
        """ Create a population and mutate it according to a mutation
        probability. Verify that the mutated population is of the correct size,
        and that each element of the population is an individual. """

        pdfs = [Gamma, Poisson]
        for pdf in pdfs:
            pdf.alt_pdfs = [p for p in pdfs if p != pdf]

        population = create_initial_population(size, row_limits, col_limits,
                                               pdfs, weights)
        mutant_population = mutate_population(population, mutation_prob,
                                              allele_prob, row_limits,
                                              col_limits, pdfs, weights)
        assert isinstance(mutant_population, list)
        assert len(mutant_population) == len(population)

        for ind in mutant_population:
            assert len(ind) == ind[1] + 2
            assert isinstance(ind[0], int) and isinstance(ind[1], int)

            for col in ind[2:]:
                assert isinstance(col, tuple(pdfs))
