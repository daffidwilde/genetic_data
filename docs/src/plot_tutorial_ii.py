import numpy as np
import matplotlib.pyplot as plt
import edo


def squared_error(df, num_samples=50):

    errors = []
    for _ in range(num_samples):
        sample_mean = df.sample(5, replace=True).mean().mean()
        errors.append((df.mean().mean() - sample_mean) ** 2)
    return sum(errors) / num_samples


pop, fit, all_pops, all_fits = edo.run_algorithm(
    fitness=squared_error,
    size=100,
    row_limits=[5, 50],
    col_limits=[1, 1],
    max_iter=25,
    seed=0,
)

fig, ax = plt.subplots(1, figsize=(32, 12), dpi=300)

ax.boxplot(all_fits, positions=range(len(all_fits)), sym=".", showmeans=True)

ax.set_title("Fitness scores in each epoch", size=24, pad=25)
ax.set_yscale("log")
ax.set_xlabel("Epoch", size=24)
ax.set_ylabel(r"$\log(f(X))$", size=24)

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(20)

plt.tight_layout()
plt.savefig("../_static/tutorial_ii_plot.png")
