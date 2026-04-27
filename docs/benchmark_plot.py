# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "msprime>=1.3",
#     "numpy",
#     "matplotlib",
#     "SciencePlots",
# ]
# ///
# NOTE: smc_prime must be installed separately (`make dev` from repo root)

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    import scienceplots
    import msprime
    import smc_prime
    from time import perf_counter

    plt.style.use("science")
    return mo, msprime, np, perf_counter, plt, smc_prime


@app.cell
def _(mo):
    mo.md(r"""
    # SMC' Benchmark: Runtime vs Population Size

    Wall-clock time per replicate for three coalescent algorithms
    across a range of effective population sizes $N_e$.
    """)
    return


@app.cell
def _(msprime, np, perf_counter, smc_prime):
    # --- Parameters ---
    Ne_values = np.array([10, 50, 100, 500, 1_000, 5_000, 10_000, 100_000])
    sample_sizes = [10, 100]
    n_reps = 20
    L = 1.0  # 1 Morgan
    r = 1.0  # recombination rate (per Morgan)
    seeds = list(range(1, n_reps + 1))

    # --- Timing helpers ---
    def time_smc_prime(Ne, n):
        t0 = perf_counter()
        for s in seeds:
            smc_prime.sim_ancestry(
                population_size=float(Ne),
                num_samples=n,
                sequence_length=L,
                recombination_rate=r,
                random_seed=s,
            )
        return (perf_counter() - t0) / n_reps

    def time_msprime(Ne, n, model):
        t0 = perf_counter()
        for s in seeds:
            msprime.sim_ancestry(
                samples=n,
                ploidy=1,
                population_size=Ne,
                model=model,
                sequence_length=L/1e-8,
                recombination_rate=1e-8,
                random_seed=s,
            )
        return (perf_counter() - t0) / n_reps

    # --- Run benchmarks ---
    # Max Ne cutoff per algorithm (hudson is O(N) so cap it)
    max_Ne = {
        "smc_prime (Rust)": np.inf,
        "msprime hudson": 10_000,
        "msprime smc_prime": np.inf,
    }
    algorithms = {
        "smc_prime (Rust)": lambda Ne, n: time_smc_prime(Ne, n),
        "msprime hudson": lambda Ne, n: time_msprime(Ne, n, "hudson"),
        "msprime smc_prime": lambda Ne, n: time_msprime(Ne, n, msprime.SMCK(k=1)),
    }

    results = {}  # (algo, n) -> (Ne_arr, times_arr)
    for algo_name, fn in algorithms.items():
        _ne_subset = Ne_values[Ne_values <= max_Ne[algo_name]]
        for n in sample_sizes:
            key = (algo_name, n)
            times = []
            for Ne in _ne_subset:
                times.append(fn(Ne, n))
                print(f"{algo_name}  n={n}  Ne={Ne}  {times[-1]*1000:.1f} ms/rep")
            results[key] = (_ne_subset, np.array(times))
    return L, algorithms, results, sample_sizes


@app.cell
def _(L, algorithms, plt, results, sample_sizes):
    # --- Plot ---
    colors = {
        "smc_prime (Rust)": "C0",
        "msprime hudson": "C1",
        "msprime smc_prime": "C2",
    }

    labels = {
        "smc_prime (Rust)": "SMC' (this library)",
        "msprime hudson": "Hudson (msprime)",
        "msprime smc_prime": "SMC' (msprime)",
    }

    plt.rc("figure", autolayout=True)
    ONE_MM = 1 / 25.4  # Convert mm to inches
    fig, axes = plt.subplots(1, 2, figsize=(85 * ONE_MM * 2, 70 * ONE_MM), sharey=True)

    for i, _n in enumerate(sample_sizes):
        ax = axes[i]
        for _algo_name in algorithms:
            _ne, _t = results[(_algo_name, _n)]
            ax.plot(
                _ne,
                _t,
                color=colors[_algo_name],
                marker="o",
                markersize=4,
                label=labels[_algo_name],
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Effective population size $N$")
        ax.set_title(rf"$n={_n}$ lineages, $L={L:.0f}$ Morgan")
        if i == 0:
            ax.set_ylabel("Time per replicate (seconds)")
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("docs/benchmark_plot.svg", dpi=200)
    fig
    return


if __name__ == "__main__":
    app.run()
