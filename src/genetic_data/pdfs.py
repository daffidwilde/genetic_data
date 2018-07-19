""" Default column probability density functions for the genetic algorithm. """

import random
from scipy.stats import gamma, poisson
import numpy as np

class Gamma():
    """ Continuous column pdf given by the gamma distribution. """

    def __init__(self):

        self.alpha = random.uniform(0, 100)
        self.theta = random.uniform(0, 100)
        self.seed = random.randint(0, 1e6)

    def __str__(self):
        return 'Gamma'

    def get_params(self):
        """ Return current distribution parameters. """
        return tuple([self.alpha, self.theta])

    def sample(self, nrows, seed):
        """ Take a sample of size `nrows` from the gamma distribution with
        parameters `alpha` and `theta`. Seeded for reproducibility. """

        np.random.seed(seed)
        return gamma.rvs(a=self.alpha, scale=self.theta, size=nrows)

    def mutate(self, change_pdf=False, change_alpha=False,
               change_theta=False):
        """ Mutation of the column. This is either changing to another pdf all
        together, or a change in the current parameters.

        Parameters
        ----------
        change_pdf : bool
            Change to another pdf. If so, no parameter mutation necessary.
        change_alpha : bool)
            Mutate value of alpha, i.e. the shape of the distribution.
        change_theta : bool
            Mutate the value of theta, i.e. the scale of the distribution.

        Returns
        -------
        self : object
            A mutated Gamma object.
        """

        if change_pdf and self.alt_pdfs:
            self = random.choice(self.alt_pdfs)()
            return self
        if change_alpha:
            old_alpha = self.alpha
            self.alpha = random.uniform(0, 100)
            while self.alpha in [0, old_alpha]:
                self.alpha = random.uniform(0, 100)
        if change_theta:
            old_theta = self.theta
            self.theta = random.uniform(0, 100)
            while self.theta in [0, old_theta]:
                self.theta = random.uniform(0, 100)
        return self

class Poisson():
    """ Discrete column pdf given by the Poisson distribution. """

    def __init__(self):

        self.mu = random.uniform(0, 100)
        self.seed = random.randint(0, 1e6)

    def __str__(self):
        return 'Poisson'

    def get_params(self):
        """ Return current distribution parameters. """
        return tuple([self.mu])

    def sample(self, nrows, seed):
        """ Take a sample of size `nrows` from the Poisson distribution with
        parameter `mu`. Seeded for reproducibility. """

        np.random.seed(seed)
        return poisson.rvs(self.mu, size=nrows)

    def mutate(self, change_pdf=False, change_mu=False):
        """ Mutation of the column. This is either changing to another pdf all
        together, or a change in the current parameter.

        Parameters
        ----------
        change_pdf : bool
            Change to another pdf. If so, no parameter mutation necessary.
        change_mu : bool
            Mutate the value of mu, i.e. the mean of the distribution.

        Returns
        -------
        self : object
            A mutated Gamma object.
        """

        if change_pdf and self.alt_pdfs:
            self = random.choice(self.alt_pdfs)()
            return self
        if change_mu:
            old_mu = self.mu
            self.mu = random.uniform(0, 100)
            while self.mu == old_mu:
                self.mu = random.uniform(0, 100)
        return self
