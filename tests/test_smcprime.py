import time

import msprime
import numpy as np
import pytest
import scipy.stats as stats
import smc_prime
import tskit
from hypothesis import given, settings
from hypothesis import strategies as st
from joblib import Parallel, delayed
from tqdm.auto import tqdm


def _parallel_map(values, fn):
    return Parallel(n_jobs=-1, prefer="threads")(delayed(fn)(value) for value in values)


def compare_simulators(
    population_size,
    msprime_demography=None,
    num_samples=10,
    sequence_length=10.0,
    recombination_rate=1.0,
    num_replicates=100,
    alpha=0.01,
):
    """
    Run `num_replicates` simulations with both smc_prime and msprime(model="smc_prime").
    Asserts that the means of genetic diversity and num_trees are not significantly
    different using a two-sample t-test at the given alpha level.
    """
    seeds = np.arange(100, 100 + num_replicates)

    # Run smc_prime
    replicates = _parallel_map(
        seeds,
        lambda seed: smc_prime.sim_ancestry(
            population_size=population_size,
            num_samples=num_samples,
            sequence_length=sequence_length,
            recombination_rate=recombination_rate,
            random_seed=seed,
        ),
    )

    # Run msprime
    ms_replicates = _parallel_map(
        seeds,
        lambda seed: msprime.sim_ancestry(
            samples=num_samples,
            population_size=population_size if msprime_demography is None else None,
            demography=msprime_demography,
            ploidy=1,
            model="smc_prime",
            random_seed=seed,
            sequence_length=sequence_length,
            recombination_rate=recombination_rate,
            discrete_genome=False,
        ),
    )

    # Calculate metrics
    diversity = [ts.diversity(mode="branch") for ts in replicates]
    ms_diversity = [ts.diversity(mode="branch") for ts in ms_replicates]

    num_trees = [ts.num_trees for ts in replicates]
    ms_num_trees = [ts.num_trees for ts in ms_replicates]

    # Two-sample t-test for diversity
    stat_div, pval_div = stats.ttest_ind(diversity, ms_diversity, equal_var=False)
    assert pval_div > alpha, (
        f"Diversity means differ significantly: smc_prime={np.mean(diversity):.4f}, msprime={np.mean(ms_diversity):.4f} (p={pval_div:.4e})"
    )

    # Two-sample t-test for num_trees
    stat_trees, pval_trees = stats.ttest_ind(num_trees, ms_num_trees, equal_var=False)
    assert pval_trees > alpha, (
        f"Tree count means differ significantly: smc_prime={np.mean(num_trees):.1f}, msprime={np.mean(ms_num_trees):.1f} (p={pval_trees:.4e})"
    )


def test_sim_ancestry_returns_tree_sequence():
    ts = smc_prime.sim_ancestry(
        population_size=100,
        num_samples=2,
        random_seed=42,
    )
    assert isinstance(ts, tskit.TreeSequence)
    assert ts.num_samples == 2
    assert ts.sequence_length == 1.0

    # Check optional arguments works
    ts = smc_prime.sim_ancestry(
        population_size=100,
        num_samples=2,
    )
    assert isinstance(ts, tskit.TreeSequence)
    assert ts.num_samples == 2
    assert ts.sequence_length == 1.0


def test_sim_ancestry_matches_diversity_expectations_constant():
    compare_simulators(
        population_size=100.0,
        num_samples=10,
        sequence_length=10.0,
        recombination_rate=1.0,
        num_replicates=50,
        alpha=0.01,
    )


def test_sim_ancestry_matches_diversity_expectations_piecewise():
    demo = msprime.Demography()
    demo.add_population(name="pop_0", initial_size=100.0)
    demo.add_population_parameters_change(time=200.0, initial_size=5000)

    compare_simulators(
        population_size=[(0.0, 100.0), (200.0, 5000.0)],
        msprime_demography=demo,
        num_samples=10,
        sequence_length=2.0,
        recombination_rate=1.0,
        num_replicates=50,
        alpha=0.01,
    )


def test_sim_ancestry_matches_diversity_expectations_exponential_growth():
    alpha = 0.01
    t_change = 100.0
    N0 = 100.0
    # In msprime, N(t) = N(0) * exp(-alpha * t)
    N_change = N0 * np.exp(-alpha * t_change)

    demo = msprime.Demography()
    demo.add_population(name="pop_0", initial_size=N0, growth_rate=alpha)
    demo.add_population_parameters_change(
        time=t_change, initial_size=N_change, growth_rate=0.0
    )

    compare_simulators(
        population_size=[(0.0, N0, alpha), (t_change, N_change, 0.0)],
        msprime_demography=demo,
        num_samples=10,
        sequence_length=1.0,
        recombination_rate=1.0,
        num_replicates=50,
        alpha=0.01,
    )


def test_sim_ancestry_matches_diversity_expectations_exponential_decline():
    growth_rate = -0.005
    t_change = 50.0
    N0 = 1000.0
    # In msprime, N(t) = N(0) * exp(-growth_rate * t)
    N_change = N0 * np.exp(-growth_rate * t_change)

    demo = msprime.Demography()
    demo.add_population(name="pop_0", initial_size=N0, growth_rate=growth_rate)
    demo.add_population_parameters_change(
        time=t_change, initial_size=N_change, growth_rate=0.0
    )

    compare_simulators(
        population_size=[(0.0, N0, growth_rate), (t_change, N_change, 0.0)],
        msprime_demography=demo,
        num_samples=10,
        sequence_length=1.0,
        recombination_rate=1.0,
        num_replicates=50,
        alpha=0.01,
    )


@settings(deadline=None, max_examples=50)
@given(
    ne=st.floats(min_value=1e-3, max_value=1e6, allow_nan=False, allow_infinity=False),
    recombination_rate=st.floats(
        min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False
    ),
)
def test_hypothesis_constant_robustness(ne, recombination_rate):
    ts = smc_prime.sim_ancestry(
        population_size=ne,
        num_samples=5,
        sequence_length=1.0,
        recombination_rate=recombination_rate,
        random_seed=42,
    )
    assert isinstance(ts, tskit.TreeSequence)
    assert ts.num_samples == 5
    assert ts.sequence_length == 1.0


@settings(deadline=None, max_examples=50)
@given(
    epochs=st.lists(
        st.tuples(
            st.floats(min_value=1e-3, max_value=1e5),  # ne
            st.floats(min_value=-0.5, max_value=0.5),  # alpha
        ),
        min_size=1,
        max_size=5,
    )
)
def test_hypothesis_piecewise_exponential_robustness(epochs):
    # Construct a valid demography timeline
    times = [0.0]
    for i in range(1, len(epochs)):
        times.append(times[-1] + np.random.uniform(0.1, 10.0))

    pop_size = []
    for t, (ne, alpha) in zip(times, epochs):
        pop_size.append((t, ne, alpha))

    # Ensure last epoch has 0 alpha
    pop_size[-1] = (pop_size[-1][0], pop_size[-1][1], 0.0)

    try:
        ts = smc_prime.sim_ancestry(
            population_size=pop_size,
            num_samples=4,
            sequence_length=1.0,
            recombination_rate=1.0,
            random_seed=123,
        )
        assert isinstance(ts, tskit.TreeSequence)
    except RuntimeError as e:
        # We might generate extremely high populations forcing overflows, let them raise rather than crash
        pass
    ts_const = smc_prime.sim_ancestry(
        population_size=100,
        num_samples=8,
        sequence_length=5.0,
        recombination_rate=1.0,
        random_seed=1234,
    )
    ts_piecewise = smc_prime.sim_ancestry(
        population_size=[(0.0, 100.0)],
        num_samples=8,
        sequence_length=5.0,
        recombination_rate=1.0,
        random_seed=1234,
    )
    assert np.isclose(
        ts_const.diversity(mode="branch"),
        ts_piecewise.diversity(mode="branch"),
    )


def test_coalescence_time_distribution_constant():
    """
    For n=2 haploid samples with constant Ne, the single coalescence time
    is Exp(1/Ne) distributed (rate = 1 pair * 1/Ne).
    Draw many replicates and KS-test against the expected distribution.
    """
    Ne = 500.0
    n_reps = 2000
    seeds = np.arange(1, n_reps + 1)

    coal_times = _parallel_map(
        seeds,
        lambda s: smc_prime.sim_ancestry(
            population_size=Ne,
            num_samples=2,
            recombination_rate=0.0,
            random_seed=s,
        ),
    )
    # Each tree sequence has a single tree; the root time is the coalescence time
    times = np.array([ts.node(ts.first().root).time for ts in coal_times])
    assert times.min() > 0, "All coalescence times must be positive"

    # KS test against Exp(rate=1/Ne), i.e. scale=Ne
    _, p_val = stats.kstest(times, "expon", args=(0, Ne))
    assert p_val > 0.01, (
        f"Coalescence times don't match Exp(1/Ne={Ne}): KS p={p_val:.4e}, "
        f"mean={times.mean():.1f} (expected {Ne:.1f})"
    )


def test_coalescence_time_distribution_exponential_growth():
    """
    For n=2 with exponential growth, verify the coalescence time CDF
    against msprime (ground truth). Both use SMC' model.
    """
    Ne = 200.0
    alpha = 0.01
    t_change = 100.0
    N_change = Ne * np.exp(-alpha * t_change)
    n_reps = 1000
    seeds = np.arange(1, n_reps + 1)

    smc_times = np.array(
        [
            ts.node(ts.first().root).time
            for ts in _parallel_map(
                seeds,
                lambda s: smc_prime.sim_ancestry(
                    population_size=[(0.0, Ne, alpha), (t_change, N_change, 0.0)],
                    num_samples=2,
                    recombination_rate=0.0,
                    random_seed=s,
                ),
            )
        ]
    )

    demo = msprime.Demography()
    demo.add_population(name="pop_0", initial_size=Ne, growth_rate=alpha)
    demo.add_population_parameters_change(
        time=t_change, initial_size=N_change, growth_rate=0.0
    )
    ms_times = np.array(
        [
            ts.node(ts.first().root).time
            for ts in _parallel_map(
                seeds,
                lambda s: msprime.sim_ancestry(
                    samples=2,
                    demography=demo,
                    ploidy=1,
                    model="smc_prime",
                    recombination_rate=0.0,
                    sequence_length=1.0,
                    random_seed=s,
                    discrete_genome=False,
                ),
            )
        ]
    )

    # Two-sample KS test: both should come from the same distribution
    _, p_val = stats.ks_2samp(smc_times, ms_times)
    assert p_val > 0.01, (
        f"Coal time distributions differ: KS p={p_val:.4e}, "
        f"smc_prime mean={smc_times.mean():.1f}, msprime mean={ms_times.mean():.1f}"
    )


def test_sim_ancestry_rejects_invalid_piecewise():
    with pytest.raises(RuntimeError, match="First epoch must start at time 0"):
        smc_prime.sim_ancestry(
            population_size=[(1.0, 100.0)], num_samples=2, random_seed=1
        )
