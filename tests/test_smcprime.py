import time

import msprime
import numpy as np
import smc_prime
import tskit
from joblib import Parallel, delayed
from tqdm.auto import tqdm


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
    replicates = [
        smc_prime.sim_ancestry(
            population_size=100,
            random_seed=seed,
            num_samples=10,
            sequence_length=10.0,
            recombination_rate=1.0,
        )
        for seed in tqdm(seeds)
    ]
    ms_replicates = [
        msprime.sim_ancestry(
            samples=10,
            population_size=100,
            ploidy=1,
            model="smc_prime",
            random_seed=seed,
            sequence_length=10.0,
            recombination_rate=1.0,
            discrete_genome=False,
        )
        for seed in tqdm(seeds)
    ]
    diversity = [ts.diversity(mode="branch") / 2 for ts in replicates]
    ms_diversity = [ts.diversity(mode="branch") / 2 for ts in ms_replicates]
    expected = 100
    assert (
        np.mean(diversity) - np.std(diversity) * 0.5
        < expected
        < np.mean(diversity) + np.std(diversity) * 0.5
    )
    assert np.abs((np.mean(ms_diversity) - np.mean(diversity))) < np.std(diversity)
    assert np.abs((np.mean(ms_diversity) - np.mean(diversity))) < np.std(ms_diversity)

    num_trees = [ts.num_trees for ts in replicates]
    ms_num_trees = [ts.num_trees for ts in ms_replicates]
    assert np.abs((np.mean(ms_num_trees) - np.mean(num_trees))) < np.std(num_trees)
    assert np.abs((np.mean(ms_num_trees) - np.mean(num_trees))) < np.std(ms_num_trees)
