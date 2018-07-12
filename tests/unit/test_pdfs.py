""" Unit tests for the standard columns pdf's. """

from copy import deepcopy
from hypothesis import given
from hypothesis.strategies import integers
from genetic_data.pdfs import Gamma, Poisson

from test_util.trivials import TrivialPDF

class TestGamma():
    """ A class containing the tests for the Gamma column pdf. """

    Gamma.alt_pdfs = [TrivialPDF]

    @given(nrows=integers(min_value=1))
    def test_sample(self, nrows):
        """ Verify that a given Gamma object can sample correctly. """
        gamma_pdf = Gamma()
        sample = gamma_pdf.sample(nrows)
        assert sample.shape == (nrows,)
        assert sample.dtype == 'float'

    def test_mutation_defaults(self):
        """ Verify that .mutate() returns self by default. """
        gamma = Gamma()
        mutant = deepcopy(gamma).mutate()
        assert mutant.alpha == gamma.alpha
        assert mutant.theta == gamma.theta
        assert mutant.seed == gamma.seed
        assert mutant.alt_pdfs == gamma.alt_pdfs

    def test_change_parameters(self):
        """ Verify Gamma object can mutate by changing its parameters. """
        gamma = Gamma()
        mutant = deepcopy(gamma).mutate(change_alpha=True, change_theta=True)
        assert mutant.alpha > 0
        assert mutant.alpha != gamma.alpha
        assert mutant.theta > 0
        assert mutant.theta != gamma.theta

    def test_change_pdf(self):
        """ Verify Gamma object can mutate to another kind of pdf. Here the
        TrivialPDF class is used for illustration. """
        gamma = Gamma()
        mutant = deepcopy(gamma).mutate(change_pdf=True)
        assert mutant != gamma
        assert 'TrivialPDF' in str(mutant)

    def test_change_seed(self):
        """ Verify Gamma object can mutate by changing its seed. """
        gamma = Gamma()
        mutant = deepcopy(gamma).mutate(change_seed=True)
        assert mutant.seed != gamma.seed


class TestPoisson():
    """ A class containing the tests for the Poisson column pdf. """

    Poisson.alt_pdfs = [TrivialPDF]

    @given(nrows=integers(min_value=1))
    def test_sample(self, nrows):
        """ Verify that a given Poisson object can sample correctly. """
        poisson = Poisson()
        sample = poisson.sample(nrows)
        assert sample.shape == (nrows,)
        assert sample.dtype == 'int'

    def test_mutation_defaults(self):
        """ Verify .mutate() returns self by default. """
        poisson = Poisson()
        mutant = deepcopy(poisson).mutate()
        assert mutant.mu == poisson.mu
        assert mutant.alt_pdfs == poisson.alt_pdfs
        assert mutant.seed == poisson.seed

    def test_change_parameter(self):
        """ Verify Poisson object can mutate by changing its parameter. """
        poisson = Poisson()
        mutant = deepcopy(poisson).mutate(change_mu=True)
        assert mutant.mu > 0
        assert mutant.mu != poisson.mu

    def test_change_pdf(self):
        """ Verify Poisson object can mutate to another kind of pdf. Here the
        TrivialPDF class is used for illustration. """
        poisson = Poisson()
        mutant = poisson.mutate(change_pdf=True)
        assert 'TrivialPDF' in str(mutant)

    def test_change_seed(self):
        """ Verify Poisson object can mutate by changing its seed. """
        poisson = Poisson()
        mutant = deepcopy(poisson).mutate(change_seed=True)
        assert mutant.seed != poisson.seed
