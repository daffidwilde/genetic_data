""" Tests for the compacting of the search space. """

import pytest

from edo.compact import compact_search_space
from edo.fitness import get_fitness
from edo.operators import selection
from edo.pdfs import Gamma, Normal, Poisson
from edo.population import create_initial_population

from .util.parameters import COMPACT_SPACE
from .util.trivials import trivial_fitness


@COMPACT_SPACE
def test_compact_search_space(
    size, row_limits, col_limits, weights, props, maximise, compact_ratio, itr
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

    if compact_ratio in [0, 1]:
        with pytest.raises(ValueError):
            compact_search_space(parents, pdfs, itr, max_iter, compact_ratio)

    else:
        compacted_pdfs = compact_search_space(
            parents, pdfs, itr, max_iter, compact_ratio
        )

        for pdf in compacted_pdfs:
            assert pdf.param_limits.keys() == vars(pdf()).keys()

            if not pdf.param_limits == pdf.hard_limits:
                for name, limits in pdf.param_limits.items():
                    assert (
                        limits[0] >= pdf.hard_limits[name][0]
                        and limits[1] <= pdf.hard_limits[name][1]
                    )
            pdf.reset()
