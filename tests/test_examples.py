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
    """A Uniform child for capturing the radius of a point in the unit
    circle."""

    name = "RadiusUniform"
    param_limits = {"bounds": [0, 1]}


class AngleUniform(Uniform):
    """A Uniform child for capturing the angle of a point in the unit
    circle."""

    name = "AngleUniform"
    param_limits = {"bounds": [-2 * np.pi, 2 * np.pi]}


def circle_fitness(individual):
    """ Determine the similarity of the dataframe to the unit circle. """

    columns = [meta.name for meta in individual.metadata]
    df = individual.dataframe
    angle_idx = columns.index("AngleUniform")
    radius_idx = columns.index("RadiusUniform")

    return df[angle_idx].var() - (df[radius_idx] - 1).abs().max()


def run_circle_example():
    """ Run a smaller version of the circle example from the paper repo. """

    fit_histories = []
    for seed in range(3):

        families = [edo.Family(RadiusUniform), edo.Family(AngleUniform)]

        do = edo.DataOptimiser(
            fitness=circle_fitness,
            size=10,
            row_limits=[5, 10],
            col_limits=[(1, 1), (1, 1)],
            families=families,
            max_iter=3,
            best_prop=0.1,
            mutation_prob=0.01,
            maximise=True,
        )

        _, fits = do.run(random_state=seed)

        fits["seed"] = seed
        fit_histories.append(fits)

    fit_history = pd.concat(fit_histories)
    return fit_history


@given(repetitions=integers())
@settings(deadline=None, max_examples=10)
def test_circle_example(repetitions):
    """Check that the circle example dataset never changes. If, for some
    reason, it changes (or needs to be changed) on purpose, then delete
    `circle.csv` and run this test again."""

    history = run_circle_example()

    try:
        expected = pd.read_csv("tests/circle.csv")
        assert np.allclose(history.values, expected.values)

    except FileNotFoundError:
        history.to_csv("tests/circle.csv", index=False)
        assume(False)


##########
# SAMPLE #
##########


def sample_fitness(individual):
    """ Take a sample of 10% of the rows and find their mean. """

    return individual.dataframe.sample(frac=0.1, random_state=0).mean().mean()


def run_sample_example():
    """ Run the sample example used in debugging the random leak. """

    opt = edo.DataOptimiser(
        sample_fitness,
        size=100,
        row_limits=[50, 100],
        col_limits=[1, 3],
        families=[edo.Family(Uniform)],
        max_iter=10,
    )

    state = np.random.RandomState(0)
    _, fit_history = opt.run(processes=4, random_state=state)
    return fit_history


@given(repetitions=integers())
@settings(deadline=None, max_examples=5)
def test_sample_example(repetitions):
    """Check that the sample example dataset never changes (with the same
    caveat as the circle example above)."""

    history = run_sample_example()

    try:
        expected = pd.read_csv("tests/sample.csv")
        assert np.allclose(history.values, expected.values)

    except FileNotFoundError:
        history.to_csv("tests/sample.csv", index=False)
        assume(False)
