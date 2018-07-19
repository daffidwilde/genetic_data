""" Unit tests for the standard columns pdf's. """

from copy import deepcopy
from hypothesis import given
from hypothesis.strategies import integers
from genetic_data.pdfs import Gamma, Poisson

from test_util.trivials import TrivialPDF

class TestGamma():
    """ A class containing the tests for the Gamma column pdf. """

    @given(nrows=integers(min_value=1),
           seed=integers(min_value=0))
    def test_sample(self, nrows, seed):
        """ Verify that a given Gamma object can sample correctly. """
        gamma_pdf = Gamma()
        sample = gamma_pdf.sample(nrows, seed)
        assert sample.shape == (nrows,)
        assert sample.dtype == 'float'

    def test_mutation_defaults(self):
        """ Verify that .mutate() returns self by default. """
        gamma = Gamma()
        mutant = deepcopy(gamma).mutate()
        assert mutant.alpha == gamma.alpha
        assert mutant.theta == gamma.theta
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
        Gamma.alt_pdfs = [TrivialPDF]
        gamma = Gamma()
        mutant = deepcopy(gamma).mutate(change_pdf=True)
        assert mutant != gamma
        assert 'TrivialPDF' in str(mutant)


class TestPoisson():
    """ A class containing the tests for the Poisson column pdf. """

    @given(nrows=integers(min_value=1),
           seed=integers(min_value=0))
    def test_sample(self, nrows, seed):
        """ Verify that a given Poisson object can sample correctly. """
        poisson = Poisson()
        sample = poisson.sample(nrows, seed)
        assert sample.shape == (nrows,)
        assert sample.dtype == 'int'

    def test_mutation_defaults(self):
        """ Verify .mutate() returns self by default. """
        poisson = Poisson()
        mutant = deepcopy(poisson).mutate()
        assert mutant.mu == poisson.mu
        assert mutant.alt_pdfs == poisson.alt_pdfs

    def test_change_parameter(self):
        """ Verify Poisson object can mutate by changing its parameter. """
        poisson = Poisson()
        mutant = deepcopy(poisson).mutate(change_mu=True)
        assert mutant.mu > 0
        assert mutant.mu != poisson.mu

    def test_change_pdf(self):
        """ Verify Poisson object can mutate to another kind of pdf. Here the
        TrivialPDF class is used for illustration. """

        Poisson.alt_pdfs = [TrivialPDF]
        poisson = Poisson()
        mutant = poisson.mutate(change_pdf=True)
        assert 'TrivialPDF' in str(mutant)
