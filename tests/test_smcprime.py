import time

import msprime
import numpy as np
import pytest
import smc_prime
import tskit
from joblib import Parallel, delayed
from tqdm.auto import tqdm


def _parallel_map(values, fn):
    return Parallel(n_jobs=-1, prefer="threads")(delayed(fn)(value) for value in values)


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


def test_sim_ancestry_matches_diversity_expectations():
    seeds = np.arange(100, 200)
    replicates = _parallel_map(
        tqdm(seeds),
        lambda seed: smc_prime.sim_ancestry(
            population_size=100,
            random_seed=seed,
            num_samples=10,
            sequence_length=10.0,
            recombination_rate=1.0,
        ),
    )
    ms_replicates = _parallel_map(
        seeds,
        lambda seed: msprime.sim_ancestry(
            samples=10,
            population_size=100,
            ploidy=1,
            model="smc_prime",
            random_seed=seed,
            sequence_length=10.0,
            recombination_rate=1.0,
            discrete_genome=False,
        ),
    )
    diversity = _parallel_map(replicates, lambda ts: ts.diversity(mode="branch") / 2)
    ms_diversity = _parallel_map(
        ms_replicates, lambda ts: ts.diversity(mode="branch") / 2
    )
    expected = 100
    assert (
        np.mean(diversity) - np.std(diversity) * 0.5
        < expected
        < np.mean(diversity) + np.std(diversity) * 0.5
    )
    assert np.abs((np.mean(ms_diversity) - np.mean(diversity))) < np.std(diversity)
    assert np.abs((np.mean(ms_diversity) - np.mean(diversity))) < np.std(ms_diversity)

    num_trees = _parallel_map(replicates, lambda ts: ts.num_trees)
    ms_num_trees = _parallel_map(ms_replicates, lambda ts: ts.num_trees)
    assert np.abs((np.mean(ms_num_trees) - np.mean(num_trees))) < np.std(num_trees)
    assert np.abs((np.mean(ms_num_trees) - np.mean(num_trees))) < np.std(ms_num_trees)


def test_sim_ancestry_matches_diversity_expectations_piecewise():
    seeds = np.arange(100, 120)
    replicates = _parallel_map(
        tqdm(seeds),
        lambda seed: smc_prime.sim_ancestry(
            population_size=[(0.0, 100.0), (200.0, 5000)],
            random_seed=seed,
            num_samples=10,
            sequence_length=2.0,
            recombination_rate=1.0,
        ),
    )
    demo = msprime.Demography()
    demo.add_population(name="pop_0", initial_size=100.0)
    demo.add_population_parameters_change(time=200.0, initial_size=5000)

    ms_replicates = _parallel_map(
        seeds,
        lambda seed: msprime.sim_ancestry(
            samples=10,
            ploidy=1,
            model="smc_prime",
            random_seed=seed,
            demography=demo,
            sequence_length=2.0,
            recombination_rate=1.0,
            discrete_genome=False,
        ),
    )
    diversity = _parallel_map(replicates, lambda ts: ts.diversity(mode="branch") / 2)
    ms_diversity = _parallel_map(
        ms_replicates, lambda ts: ts.diversity(mode="branch") / 2
    )
    assert np.abs((np.mean(ms_diversity) - np.mean(diversity))) < np.std(diversity)
    assert np.abs((np.mean(ms_diversity) - np.mean(diversity))) < np.std(ms_diversity)

    num_trees = _parallel_map(replicates, lambda ts: ts.num_trees)
    ms_num_trees = _parallel_map(ms_replicates, lambda ts: ts.num_trees)
    assert np.abs((np.mean(ms_num_trees) - np.mean(num_trees))) < np.std(num_trees)
    assert np.abs((np.mean(ms_num_trees) - np.mean(num_trees))) < np.std(ms_num_trees)


def test_sim_ancestry_accepts_piecewise_haploid_ne():
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


def test_sim_ancestry_rejects_invalid_piecewise_haploid_ne():
    with pytest.raises(RuntimeError, match="First epoch must start at time 0"):
        smc_prime.sim_ancestry(
            population_size=[(1.0, 100.0)], num_samples=2, random_seed=1
        )
