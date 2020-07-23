""" Some example tests. """

import numpy as np
import pandas as pd
from hypothesis import assume, given, settings
from hypothesis.strategies import integers

import edo
from edo.distributions import Uniform

##########
# CIRCLE #
##########


class RadiusUniform(Uniform):
    """ A Uniform child for capturing the radius of a point in the unit
    circle. """

    name = "RadiusUniform"
    dtype = "float"
    param_limits = {}

    def __init__(self):

        lower = np.random.uniform(*self.param_limits["bounds"])
        upper = np.random.uniform(lower, max(self.param_limits["bounds"]))
        self.bounds = [lower, upper]


class AngleUniform(Uniform):
    """ A Uniform child for capturing the angle of a point in the unit
    circle. """

    name = "AngleUniform"
    dtype = "float"
    param_limits = {}

    def __init__(self):

        lower = np.random.uniform(*self.param_limits["bounds"])
        upper = np.random.uniform(lower, max(self.param_limits["bounds"]))
        self.bounds = [lower, upper]


def circle_fitness(df):
    """ Determine the similarity of the dataframe to the unit circle. """

    return max(
        df[0].var() - (df[1] - 1).abs().max(),
        df[1].var() - (df[0] - 1).abs().max(),
    )


def split_dataframe(individual):

    df, metadata = individual
    distributions = [m["name"] for m in metadata]
    radii = df[distributions.index("RadiusUniform")]
    angles = df[distributions.index("AngleUniform")]

    return radii, angles


def run_circle_example():
    """ Run a smaller version of the circle example from the paper repo. """

    fit_histories = []
    for seed in range(3):

        RadiusUniform.param_limits["bounds"] = [0, 1]
        AngleUniform.param_limits["bounds"] = [-2 * np.pi, 2 * np.pi]

        do = edo.DataOptimiser(
            fitness=circle_fitness,
            size=10,
            row_limits=[5, 10],
            col_limits=[(1, 1), (1, 1)],
            distributions=[RadiusUniform, AngleUniform],
            max_iter=3,
            best_prop=0.1,
            mutation_prob=0.01,
            maximise=True,
        )

        _, fits = do.run(seed=seed)

        fits["seed"] = seed
        fit_histories.append(fits)

    fit_history = pd.concat(fit_histories)
    return fit_history


@given(repetitions=integers())
@settings(deadline=None, max_examples=10)
def test_circle_example(repetitions):
    """ Check that the circle example dataset never changes. If, for some
    reason, it changes (or needs to be changed) on purpose, then delete
    `circle.csv` and run this test again. """

    history = run_circle_example()

    print("\n", history)

    try:
        expected = pd.read_csv("tests/circle.csv")
        assert np.allclose(history.values, expected.values, 1e-5)

    except FileNotFoundError:
        history.to_csv("tests/circle.csv", index=False)
        assume(False)
