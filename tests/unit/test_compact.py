""" Tests for the compacting of the search space. """

import pytest

from edo.compact import compact_search_space
from edo.fitness import get_fitness
from edo.operators import selection
from edo.population import create_initial_population
from edo.pdfs import Gamma, Normal, Poisson

from .util.parameters import COMPACT_SPACE
from .util.trivials import trivial_fitness


@COMPACT_SPACE
def test_compact_search_space(
    size, row_limits, col_limits, weights, props, maximise, compaction_ratio,
    itr
):
    """ Test that the search space (the space of pdf parameter limits) of a
    hypothetical GA is reduced and centred around the best individuals'
    parameters at a particular iteration. """

    best_prop, lucky_prop = props
    pdfs = [Gamma, Normal, Poisson]
    max_iter = 100
    population = create_initial_population(
        size, row_limits, col_limits, pdfs, weights
    )

    pop_fitness = get_fitness(trivial_fitness, population)
    parents = selection(
        population, pop_fitness, best_prop, lucky_prop, maximise
    )

    if compaction_ratio == 1:
        with pytest.raises(ValueError):
            compacted_pdfs = compact_search_space(
                parents, pdfs, itr, max_iter, compaction_ratio
            )

    else:
        compacted_pdfs = compact_search_space(
            parents, pdfs, itr, max_iter, compaction_ratio
        )

        if compaction_ratio == 0:
            assert set(compacted_pdfs) == set(pdfs)
