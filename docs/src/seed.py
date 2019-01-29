""" .. How and why to set a seed. """

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import edo
from edo.pdfs import Normal


def x_squared(df):
    """ Squared-value fitness. """
    return df.iloc[0, 0] ** 2


def main():
    """ Run the GA with two seeds to show they produce different results. """

    pop, fit, all_pops, all_fits = edo.run_algorithm(
        x_squared,
        size=100,
        row_limits=[1, 1],
        col_limits=[1, 1],
        pdfs=[Normal],
        max_iter=5,
        seed=0,
    )

    new_pop, new_fit, new_all_pops, new_all_fits = edo.run_algorithm(
        x_squared,
        size=100,
        row_limits=[1, 1],
        col_limits=[1, 1],
        pdfs=[Normal],
        max_iter=5,
        seed=1,
    )

    with open("../how-to/seed_inds.rst", "w") as inds:
        string = ".. :orphan:\n\n"
        string += (
            "Now, by looking at the first individual in the final "
            "population for each run, we see they are different::\n\n    "
        )
        string += ">>> pop[0]\n    "
        string += str(pop[0]).replace("\n", "\n    ") + "\n    "
        string += ">>> new_pop[0]\n    "
        string += str(new_pop[0]).replace("\n", "\n    ") + "\n    "
        inds.write(string)

    fig, ax = plt.subplots(1, figsize=(32, 12), dpi=300)

    width, epsilon = 0.3, 0.01
    positions = np.arange(len(all_fits))
    shift = 0.5 * width + epsilon

    old = ax.boxplot(
        all_fits, positions=positions - shift, widths=width, patch_artist=True
    )

    new = ax.boxplot(
        new_all_fits,
        positions=positions + shift,
        widths=width,
        patch_artist=True,
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(positions)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontsize=24)
    ax.set_ylabel(r"$\log (f(x))$", fontsize=24)

    for plot, colour in zip([old, new], ["lightblue", "lightpink"]):
        for patch in plot["boxes"]:
            patch.set_facecolor(colour)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(20)

    ax.legend(
        handles=[Patch(color="lightblue"), Patch(color="lightpink")],
        labels=["Seed 0", "Seed 1"],
        fontsize=24,
    )

    plt.tight_layout()
    plt.savefig("../_static/seed.svg", format="svg", transparent=True)


if __name__ == "__main__":
    main()
