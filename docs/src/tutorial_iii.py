""" Script for plots in third tutorial. """

from scipy.stats import linregress

import matplotlib.pyplot as plt
import numpy as np

import edo
from edo.pdfs import Normal, Poisson


def determination(df):
    _, _, r, _, _ = linregress(df.iloc[:, 0].values, df.iloc[:, 1].values)
    return r ** 2


def main():
    """ Run the GA in the third tutorial and generate plots for the fitness
    progression and of the best individual in the final population. """

    pop, fit, all_pops, all_fits = edo.run_algorithm(
        fitness=determination,
        size=100,
        row_limits=[10, 50],
        col_limits=[(1, 1), (1, 1)],
        pdfs=[Normal, Poisson],
        max_iter=50,
        maximise=True,
        seed=0,
    )

    # Fitness progression
    fig, ax = plt.subplots(1, figsize=(40, 20), dpi=300)

    ax.boxplot(
        all_fits, positions=range(len(all_fits)), sym=".", showmeans=True
    )

    ax.set_title("Fitness scores in each epoch", size=24, pad=25)
    ax.set_xlabel("Epoch", size=24)
    ax.set_ylabel(r"$f(X)$", size=24)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(20)

    plt.tight_layout()
    plt.savefig(
        "../_static/tutorial_iii_plot.svg", format="svg", transparent=True
    )

    # Best final individual
    fig, ax = plt.subplots(nrows=1, figsize=(12, 8), dpi=300)

    best = np.argmax(fit)
    ind = pop[best]
    df = ind.dataframe

    ax.scatter(df.select_dtypes("int"), df.select_dtypes("float"))

    ax.set_xlabel("Discrete column")
    ax.set_ylabel("Continuous column")
    ax.annotate(
        r"$r^2$ " + f"= {np.round(fit[best], 4)}",
        xy=[2, -2],
        fontsize=20,
        bbox=dict(boxstyle="round", fc="0.9"),
    )

    plt.tight_layout()
    plt.savefig(
        "../_static/tutorial_iii_ind.svg", format="svg", transparent=True
    )


if __name__ == "__main__":
    main()
