""" Generate the plot for Example 1 in the docs. """

import matplotlib.pyplot as plt
import numpy as np

import edo


def x_squared(df):
    return df.iloc[0, 0] ** 2


def main():
    """ Run the GA in Example 1 and generate a plot of all fitness scores in
    each epoch. """

    pop, fit, all_pops, all_fits = edo.run_algorithm(
        fitness=x_squared,
        size=100,
        row_limits=[1, 1],
        col_limits=[1, 1],
        max_iter=5,
        seed=0,
    )

    fig, (top, middle, bottom) = plt.subplots(
        nrows=3, ncols=2, figsize=(30, 45), dpi=300, sharex=True, sharey=True
    )

    x_data = np.linspace(-25, 25, 101)

    for i in range(6):

        if i < 2:
            axes = top
        elif i < 4:
            axes = middle
        else:
            axes = bottom

        j = i % 2
        data = [[ind.dataframe.iloc[0, 0] for ind in all_pops[i]], all_fits[i]]

        axes[j].plot(x_data, x_data ** 2, lw=3, zorder=-1)
        axes[j].scatter(*data, s=200, color="orange")

        axes[j].set_title(f"Fitness scores in epoch {i}", size=24, pad=25)

        if i in [4, 5]:
            axes[j].set_xlabel(r"$x$", size=24)
        if i in [0, 2, 4]:
            axes[j].set_ylabel("Fitness", size=24)

        for label in axes[j].get_xticklabels() + axes[j].get_yticklabels():
            label.set_fontsize(20)

    plt.tight_layout(pad=5)
    plt.savefig("../_static/tutorial_i_plot.png")


if __name__ == "__main__":
    main()
