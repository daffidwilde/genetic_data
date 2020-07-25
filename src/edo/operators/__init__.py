""" .. All bio-operators. """

from .crossover import crossover
from .mutation import mutation
from .selection import selection
from .shrink import shrink

__all__ = ["crossover", "mutation", "selection", "shrink"]
