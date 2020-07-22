""" Top-level imports for the `edo.families` subpackage. """

from .base import Distribution
from .continuous import Gamma, Normal, Uniform
from .discrete import Bernoulli, Poisson

all_families = [Bernoulli, Gamma, Normal, Poisson, Uniform]

continuous_families = [fam for fam in all_families if fam.dtype == "float"]
discrete_families = [fam for fam in all_families if fam.dtype == "int"]

__all__ = [
    "Distribution",
    "Bernoulli",
    "Gamma",
    "Normal",
    "Poisson",
    "Uniform",
    "all_families",
    "continuous_families",
    "discrete_families",
]
