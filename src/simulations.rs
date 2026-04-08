use rand::prelude::*;
use rand_distr::{Exp, Uniform};
use std::collections::HashMap;
use thiserror::Error;
use tskit::{TableCollection, TableSortOptions, TskitError};

type Time = f64;
type NodeID = usize;
type Edge = (NodeID, NodeID);

#[derive(Debug, Error)]
pub enum SMCPrimeError {
    //#[error("Tskit error: {0}")]
    //TskitError(#[from] TskitError),
    #[error("Invalid demography: {0}")]
    InvalidDemography(String),
}

// Demographic history can be approximated with a piecewise-function,
// assuming $\lambda\ is constant within the epoch.
#[derive(Debug, Clone)]
struct Epoch {
    start_time: Time,
    lambda: f64,
}
#[derive(Debug, Clone)]
pub struct Demography {
    epochs: Vec<Epoch>,
}

impl Demography {
    fn num_epochs(&self) -> usize {
        self.epochs.len()
    }

    fn epoch_index_at(&self, t: Time) -> usize {
        match self.epochs.binary_search_by(|e| {
            e.start_time
                .partial_cmp(&t)
                .unwrap_or(std::cmp::Ordering::Less)
        }) {
            Ok(i) => i,
            Err(i) => i - 1,
        }
    }
}

impl Demography {
    pub fn constant(ne: f64) -> Result<Self, SMCPrimeError> {
        if ne <= 0.0 {
            return Err(SMCPrimeError::InvalidDemography(
                "Population size must be positive".into(),
            ));
        }
        Ok(Self {
            epochs: vec![Epoch {
                start_time: 0.0,
                lambda: 1.0 / ne,
            }],
        })
    }
}

// If we have k free lineages, we can draw waiting time
// to the next coalescence time
fn draw_coalescence_time(
    rng: &mut impl Rng,
    demography: &Demography,
    t_start: Time,
    k_lineages: f64,
) -> f64 {
    let mut e: f64 = Exp::new(1.0).expect("rate 1.0 is valid").sample(rng);

    let start_idx = demography.epoch_index_at(t_start);
    let num_epochs = demography.num_epochs();

    let mut current_start = t_start;
    for i in start_idx..num_epochs {
        let lambda_i = demography.epochs[i].lambda;
        let epoch_end = if i + 1 < num_epochs {
            demography.epochs[i + 1].start_time
        } else {
            f64::INFINITY
        };
        let budget = k_lineages * lambda_i * (epoch_end - current_start);
        if e <= budget {
            return current_start + e / (k_lineages * lambda_i);
        }
        e -= budget;
        current_start = epoch_end;
    }
    unreachable!()
}

#[derive(Debug, Clone)]
struct CoalTree {
    num_samples: usize,
    time: Vec<Time>,
    parent: Vec<Option<NodeID>>,
    children: Vec<Vec<NodeID>>,
}

/// Simulate independent coalescent tree
impl CoalTree {
    fn draw(rng: &mut impl Rng, num_samples: usize, demography: &Demography) -> Self {
        let mut time = vec![0.0; num_samples];
        time.reserve_exact(num_samples * (num_samples - 1) / 2);
        let mut parent: Vec<Option<usize>> = vec![None; num_samples];
        parent.reserve_exact(num_samples * (num_samples - 1) / 2);
        let mut children: Vec<Vec<usize>> = vec![vec![]; num_samples];

        let mut lineages: Vec<usize> = (0..num_samples).collect();
        let mut t = 0.0;
        let mut next_id = num_samples;

        while lineages.len() > 1 {
            let k = lineages.len();
            let rate_mult = (k * (k - 1) / 2) as f64; // C(k, 2)
            let coal_time = draw_coalescence_time(rng, &demography, t, rate_mult);

            // Pick 2 distinct lineages with equal probability for every unordered pair.
            // Sampling idx2 from [0, k-1) and then shifting it up when idx2 >= idx1
            // is a bias-free alternative to rejection sampling: it establishes a
            // bijection between [0, k-1) and [0, k) \ {idx1}, so every remaining
            // index is equally likely in a single draw.
            let idx1 = Uniform::new(0, k).expect("valid").sample(rng);
            let mut idx2 = Uniform::new(0, k - 1).expect("valid").sample(rng);
            if idx2 >= idx1 {
                idx2 += 1;
            }

            let lin1 = lineages[idx1];
            let lin2 = lineages[idx2];

            // Create internal node
            let new_node = next_id;
            next_id += 1;
            time.push(coal_time);
            parent.push(None);
            children.push(vec![lin1, lin2]);
            parent[lin1] = Some(new_node);
            parent[lin2] = Some(new_node);

            // Remove picked lineages, add new one
            let (lo, hi) = if idx1 < idx2 {
                (idx1, idx2)
            } else {
                (idx2, idx1)
            };
            lineages.swap_remove(hi);
            lineages.swap_remove(lo);
            lineages.push(new_node);

            t = coal_time;
        }

        CoalTree {
            num_samples,
            time,
            parent,
            children,
        }
    }
    fn total_branch_length(&self) -> f64 {
        let mut total = 0.0;
        for i in 0..self.time.len() {
            if let Some(p) = self.parent[i] {
                total += self.time[p] - self.time[i];
            }
        }
        total
    }
    /// Pick a point uniformly on the tree. Returns (node, t_recomb) where the
    /// recombination falls on the branch from `node` to its parent.
    fn uniform_point(&self, rng: &mut impl Rng) -> (NodeID, Time) {
        let total = self.total_branch_length();
        let mut remaining: f64 = Uniform::new(0.0, total).expect("valid").sample(rng);

        for i in 0..self.time.len() {
            if let Some(p) = self.parent[i] {
                let branch_len = self.time[p] - self.time[i];
                if remaining < branch_len {
                    return (i, self.time[i] + remaining);
                }
                remaining -= branch_len;
            }
        }
        unreachable!()
    }

    /// Intervals above `t_recomb` where lineages are constant
    /// Returns `(t_start, t_end, num_lineages)` triples; the last has `t_end = INF`.
    fn lineage_count_intervals_above(&self, t_recomb: f64) -> Vec<(Time, Time, usize)> {
        // Collect coalescence times of live internal nodes above t_recomb
        let mut event_times: Vec<f64> = Vec::new();
        for i in self.num_samples..self.time.len() {
            if !self.children[i].is_empty() && self.time[i] > t_recomb {
                event_times.push(self.time[i]);
            }
        }
        event_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Count lineages at t_recomb = num_samples - (live coal events at or below t_recomb)
        let mut n_lineages = self.num_samples;
        for i in self.num_samples..self.time.len() {
            if !self.children[i].is_empty() && self.time[i] <= t_recomb {
                n_lineages -= 1;
            }
        }

        let mut intervals = Vec::new();
        let mut t_start = t_recomb;
        let mut k = n_lineages;
        for &t in &event_times {
            intervals.push((t_start, t, k));
            k -= 1;
            t_start = t;
        }
        // After the MRCA there is 1 lineage extending to infinity
        intervals.push((t_start, Time::INFINITY, k));
        intervals
    }

    fn branches_at_time(&self, t: f64) -> Vec<usize> {
        let mut branches = Vec::new();
        for i in 0..self.time.len() {
            if self.time[i] >= t {
                continue;
            }
            match self.parent[i] {
                Some(p) => {
                    if self.time[p] > t {
                        branches.push(i);
                    }
                }
                None => {
                    // Root of current tree: its branch extends to infinity
                    if !self.children[i].is_empty() {
                        branches.push(i);
                    }
                }
            }
        }
        branches
    }

    /// Draw re-coalescence time and target branch on the (original) tree.
    /// The floating lineage starts at `t_recomb` and coalesces at rate
    /// `n_lineages(t) * lambda(t)`.
    fn draw_recoalescence(
        &self,
        rng: &mut impl Rng,
        demography: &Demography,
        t_recomb: f64,
    ) -> (Time, usize) {
        let intervals = self.lineage_count_intervals_above(t_recomb);
        let mut e: f64 = Exp::new(1.0).expect("1.0 is valid").sample(rng);

        for &(t_start, t_end, n_lineages) in &intervals {
            let rate_mult = n_lineages as f64;
            let mut current_t = t_start;
            while current_t < t_end {
                let ei = demography.epoch_index_at(current_t);
                let lambda = demography.epochs[ei].lambda;
                let epoch_end = if ei + 1 < demography.epochs.len() {
                    demography.epochs[ei + 1].start_time
                } else {
                    Time::INFINITY
                };
                let sub_end = t_end.min(epoch_end);
                let rate = rate_mult * lambda;
                let budget = rate * (sub_end - current_t);
                if e <= budget {
                    let t_coal = current_t + e / rate;
                    let branches = self.branches_at_time(t_coal);
                    let target =
                        branches[Uniform::new(0, branches.len()).expect("valid").sample(rng)];
                    return (t_coal, target);
                }
                e -= budget;
                current_t = sub_end;
            }
        }
        unreachable!("last interval is infinite")
    }

    // The key operation is the SPR:
    // prune `cut_node` from its parent, then regraft it at
    // `t_coal` on `target_node`'s branch (in the post-prune tree).
    // Returns `(removed_edges, added_edges)` where each edge is `(parent, child)`.
    fn spr(&mut self, cut_node: usize, target_node: usize, t_coal: f64) -> (Vec<Edge>, Vec<Edge>) {
        let c = cut_node;
        let p = self.parent[c].expect("cut_node has parent");
        let s = *self.children[p]
            .iter()
            .find(|&&x| x != c)
            .expect("sibling exists");
        let g = self.parent[p];
        let t = target_node;

        // Allocate new internal node
        let n = self.time.len();
        self.time.push(t_coal);
        self.parent.push(None);
        self.children.push(vec![]);

        let mut removed = vec![(p, c), (p, s)];
        if let Some(g_id) = g {
            removed.push((g_id, p));
        }

        let mut added;

        if t == s {
            // Regraft lands on sibling's branch (which now spans up to g after prune)
            // New node n replaces p structurally
            if let Some(g_id) = g {
                added = vec![(g_id, n), (n, s), (n, c)];
            } else {
                added = vec![(n, s), (n, c)];
            }

            // Update tree
            self.children[p].clear();
            self.parent[p] = None;
            self.children[n] = vec![s, c];
            self.parent[c] = Some(n);
            self.parent[s] = Some(n);
            self.parent[n] = g;
            if let Some(g_id) = g {
                let pos = self.children[g_id].iter().position(|&x| x == p).unwrap();
                self.children[g_id][pos] = n;
            }
        } else {
            // General case: prune p, reconnect s to g, then insert n on t's branch
            let q = self.parent[t]; // t's original parent (unaffected by prune since t != s)

            if let Some(q_id) = q {
                removed.push((q_id, t));
            }

            if let Some(g_id) = g {
                added = vec![(g_id, s)];
            } else {
                added = vec![];
            }
            if let Some(q_id) = q {
                added.push((q_id, n));
            }
            added.push((n, t));
            added.push((n, c));

            // Update tree: prune
            self.children[p].clear();
            self.parent[p] = None;
            self.parent[s] = g;
            if let Some(g_id) = g {
                let pos = self.children[g_id].iter().position(|&x| x == p).unwrap();
                self.children[g_id][pos] = s;
            }

            // Update tree: regraft
            self.children[n] = vec![t, c];
            self.parent[c] = Some(n);
            self.parent[t] = Some(n);
            self.parent[n] = q;
            if let Some(q_id) = q {
                let pos = self.children[q_id].iter().position(|&x| x == t).unwrap();
                self.children[q_id][pos] = n;
            }
        }

        (removed, added)
    }
}

pub fn sim_ancestry(
    demography: &Demography,
    num_samples: usize,
    sequence_length: f64,
    recombination_rate: f64,
    seed: u64,
) -> Result<TableCollection, TskitError> {
    // I think this is an adequate random seed generator for this purpose.
    let mut rng = rand::rngs::Xoshiro256PlusPlus::seed_from_u64(seed);
    // We initialize the ARG at x=0
    let mut x = 0.0;
    let mut tree = CoalTree::draw(&mut rng, num_samples, demography);

    // We have to keep track of "active" edges (i.e. those that go up to sequence_length)
    // Open edges: (parent_node, child_node) -> left genomic position
    let mut open_edges: HashMap<(usize, usize), f64> = HashMap::new();
    // Recorded edges: (left, right, parent_node, child_node)
    let mut edge_records: Vec<(f64, f64, usize, usize)> = Vec::new();

    // Open all initial tree edges at position 0
    for i in 0..tree.time.len() {
        if let Some(p) = tree.parent[i] {
            open_edges.insert((p, i), 0.0);
        }
    }
    while x < sequence_length {
        // This could be change to update iteratively for better performance.
        let total_bl = tree.total_branch_length();
        let rate = recombination_rate * total_bl;
        x += Exp::new(rate)
            .expect("Invalid coalescence rate")
            .sample(&mut rng);
        if x >= sequence_length {
            break;
        }
        // We pick a point in the tree uniformly
        let (cut_node, t_recomb) = tree.uniform_point(&mut rng);
        let (t_coal, target_node) = tree.draw_recoalescence(&mut rng, demography, t_recomb);
        // Healing: re-coalescence lands on the same branch that was cut
        if target_node == cut_node {
            continue;
        }
        // If target is the parent being pruned, remap to sibling (whose branch
        // absorbs p's range after the prune)
        let p = tree.parent[cut_node].unwrap();
        let effective_target = if target_node == p {
            *tree.children[p].iter().find(|&&c| c != cut_node).unwrap()
        } else {
            target_node
        };

        let (removed, added) = tree.spr(cut_node, effective_target, t_coal);
        // Close removed edges
        for &(par, chi) in &removed {
            if let Some(left) = open_edges.remove(&(par, chi)) {
                if x > left {
                    edge_records.push((left, x, par, chi));
                }
            }
        }
        // Open added edges
        for &(par, chi) in &added {
            open_edges.insert((par, chi), x);
        }
    }
    // We're done!
    // Close all remaining open edges at sequence_length
    for ((par, chi), left) in open_edges.drain() {
        if sequence_length > left {
            edge_records.push((left, sequence_length, par, chi));
        }
    }
    let mut table = TableCollection::new(sequence_length)?;
    let population = table.add_population()?;
    let defaults = tskit::NodeDefaults {
        population,
        ..Default::default()
    };
    let sample_defaults = tskit::NodeDefaults {
        flags: tskit::NodeFlags::new_sample(),
        population,
        ..Default::default()
    };

    let mut node_ids = Vec::new();
    for i in 0..tree.time.len() {
        let nid = if i < num_samples {
            table.add_node_with_defaults(0.0, &sample_defaults)?
        } else {
            table.add_node_with_defaults(tree.time[i], &defaults)?
        };
        node_ids.push(nid);
    }

    for &(left, right, par, chi) in &edge_records {
        table.add_edge(left, right, node_ids[par], node_ids[chi])?;
    }

    table.full_sort(TableSortOptions::default())?;
    table.build_index()?;
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Generate a valid piecewise-constant demography. Epoch boundaries are
    /// derived from cumulative sums of positive gaps so they are always
    /// strictly increasing and start at 0, satisfying the invariant assumed by
    /// `epoch_index_at`.
    fn arb_demography() -> impl Strategy<Value = Demography> {
        (1usize..=10usize)
            .prop_flat_map(|n| {
                let gaps = proptest::collection::vec(1.0f64..=500.0f64, n - 1);
                let lambdas = proptest::collection::vec(0.1f64..=10.0f64, n);
                (gaps, lambdas)
            })
            .prop_map(|(gaps, lambdas)| {
                let mut times = vec![0.0f64];
                for g in &gaps {
                    times.push(times.last().unwrap() + g);
                }
                Demography {
                    epochs: times
                        .into_iter()
                        .zip(lambdas)
                        .map(|(t, lambda)| Epoch {
                            start_time: t,
                            lambda,
                        })
                        .collect(),
                }
            })
    }

    proptest! {
        #[test]
        fn coal_tree_is_valid(
            seed in 0u64..u64::MAX,
            num_samples in 2usize..=50,
            demography in arb_demography(),
        ) {
            let mut rng = rand::rngs::Xoshiro256PlusPlus::seed_from_u64(seed);
            let tree = CoalTree::draw(&mut rng, num_samples, &demography);

            // A binary coalescent for n samples produces exactly 2n-1 nodes
            prop_assert_eq!(tree.time.len(), 2 * num_samples - 1);

            // Exactly one root — the MRCA
            let n_roots = tree.parent.iter().filter(|p| p.is_none()).count();
            prop_assert_eq!(n_roots, 1);

            // Every internal node is binary
            for i in num_samples..tree.time.len() {
                prop_assert_eq!(tree.children[i].len(), 2,
                    "internal node {} has {} children", i, tree.children[i].len());
            }

            // Parent times are strictly greater than child times (ultrametricity)
            for i in 0..tree.time.len() {
                if let Some(p) = tree.parent[i] {
                    prop_assert!(tree.time[p] > tree.time[i],
                        "node {i}: parent time {} <= child time {}", tree.time[p], tree.time[i]);
                }
            }
        }
        #[test]
        fn short_tree_is_valid(
            seed in 0u64..u64::MAX,
            num_samples in 2usize..=50,
            demography in arb_demography(),
        ) {
            let mut rng = rand::rngs::Xoshiro256PlusPlus::seed_from_u64(seed);
            let arg = sim_ancestry(
                &demography, num_samples, 1.0, 1.0, seed
            ).unwrap();
        }
    }
}
