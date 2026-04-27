use crate::demography::{Demography, Time};
use arrayvec::ArrayVec;
use ordered_float::NotNan;
use rand::prelude::*;
use rand_distr::{Exp, Uniform};
use std::collections::{BTreeMap, HashMap};
use std::ops::Bound::Excluded;
use tskit::{TableCollection, TableSortOptions, TskitError};

pub type NodeID = usize;
pub type Edge = (NodeID, NodeID);

/// Draw coalescence time for `k` lineages starting at `t_start`.
/// Uses fresh Exp(1) draws per epoch (memoryless property), matching msprime's approach.
fn draw_coalescence_time(
    rng: &mut impl Rng,
    demography: &Demography,
    t_start: Time,
    k: f64,
) -> f64 {
    let start_idx = demography.epoch_index_at(t_start);
    let mut current_t = t_start;
    for i in start_idx..demography.num_epochs() {
        let epoch = &demography.epochs[i];
        let epoch_end = demography.epoch_end(i);
        let e: f64 = Exp::new(1.0).expect("rate 1.0 is valid").sample(rng);
        if let Some(t_coal) = epoch.invert(current_t, e, k) {
            if t_coal <= epoch_end {
                return t_coal;
            }
        }
        current_t = epoch_end;
    }
    unreachable!()
}

#[derive(Debug, Clone)]
struct CoalTree {
    num_samples: usize,
    time: Vec<Time>,
    parent: Vec<Option<NodeID>>,
    /// `None` for leaves and dead (pruned) nodes; `Some([left, right])` for live internal nodes.
    children: Vec<Option<[NodeID; 2]>>,
    /// BTreeMap keyed by parent time (Inf for the root).
    /// Value: child node IDs whose branch ends at that parent time.
    branch_map: BTreeMap<NotNan<f64>, Vec<NodeID>>,
    total_branch_length: f64,
}

impl CoalTree {
    fn draw(rng: &mut impl Rng, num_samples: usize, demography: &Demography) -> Self {
        let mut time = vec![0.0; num_samples];
        let num_events = num_samples * (num_samples - 1) / 2;
        time.reserve_exact(num_events);
        let mut parent: Vec<Option<usize>> = vec![None; num_samples];
        parent.reserve_exact(num_events);
        let mut children: Vec<Option<[NodeID; 2]>> = vec![None; num_samples];

        let mut lineages: Vec<usize> = (0..num_samples).collect();
        let mut t = 0.0;
        let mut next_id = num_samples;

        fn sample_pair(rng: &mut impl Rng, k: usize) -> (usize, usize) {
            // Sample a pair of lineages in [0, k] without replacement
            let idx1 = Uniform::new(0, k).expect("valid").sample(rng);
            let mut idx2 = Uniform::new(0, k - 1).expect("valid").sample(rng);
            if idx2 >= idx1 {
                idx2 += 1;
            }
            (idx1, idx2)
        }

        while lineages.len() > 1 {
            let k = lineages.len();
            // Number of unique pairs
            let rate_mult = (k * (k - 1) / 2) as f64;
            // Sample coal event
            let coal_time = draw_coalescence_time(rng, demography, t, rate_mult);
            let (idx1, idx2) = sample_pair(rng, k);
            let lin1 = lineages[idx1];
            let lin2 = lineages[idx2];

            let new_node = next_id;
            next_id += 1;
            time.push(coal_time);
            parent.push(None);
            children.push(Some([lin1, lin2]));
            parent[lin1] = Some(new_node);
            parent[lin2] = Some(new_node);

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

        // Initialize branch map
        let mut branch_map: BTreeMap<NotNan<f64>, Vec<NodeID>> = BTreeMap::new();
        let mut total_branch_length = 0.0;
        for i in 0..time.len() {
            match parent[i] {
                Some(p_id) => {
                    branch_map
                        .entry(NotNan::new(time[p_id]).expect("Not NaN"))
                        .or_default()
                        .push(i);
                    total_branch_length += time[p_id] - time[i];
                }
                None => {
                    if children[i].is_some() {
                        branch_map
                            .entry(NotNan::new(f64::INFINITY).expect("Not NaN"))
                            .or_default()
                            .push(i);
                    }
                }
            }
        }

        CoalTree {
            num_samples,
            time,
            parent,
            children,
            branch_map,
            total_branch_length,
        }
    }

    fn total_branch_length(&self) -> f64 {
        self.total_branch_length
    }

    // Picks a uniform point in the "flatenned" tree and returns the affected node and the time.
    fn uniform_point(&self, rng: &mut impl Rng) -> (NodeID, Time) {
        let mut remaining: f64 = Uniform::new(0.0, self.total_branch_length)
            .expect("valid")
            .sample(rng);

        let inf_key = NotNan::new(f64::INFINITY).expect("Not nan");
        for (&death_key, children) in self.branch_map.range(..inf_key) {
            let death_time = *death_key;
            for &child in children {
                let branch_len = death_time - self.time[child];
                if remaining < branch_len {
                    return (child, self.time[child] + remaining);
                }
                remaining -= branch_len;
            }
        }
        unreachable!()
    }

    fn bmap_remove(map: &mut BTreeMap<NotNan<f64>, Vec<NodeID>>, key: NotNan<f64>, node: NodeID) {
        if let Some(v) = map.get_mut(&key) {
            if let Some(pos) = v.iter().position(|&x| x == node) {
                v.swap_remove(pos);
            }
            if v.is_empty() {
                map.remove(&key);
            }
        }
    }

    fn bmap_add(map: &mut BTreeMap<NotNan<f64>, Vec<NodeID>>, key: NotNan<f64>, node: NodeID) {
        map.entry(key).or_default().push(node);
    }

    fn nth_branch_at_time(&self, t: f64, idx: usize) -> usize {
        let t_key = NotNan::new(t).expect("Not nan");
        let mut count = 0;
        for (_, children) in self
            .branch_map
            .range((Excluded(t_key), std::ops::Bound::Unbounded))
        {
            for &child in children {
                if self.time[child] < t {
                    if count == idx {
                        return child;
                    }
                    count += 1;
                }
            }
        }
        unreachable!("idx out of range in nth_branch_at_time")
    }

    /// Draw re-coalescence time and target branch using `branch_map` directly.
    /// Each non-∞ key in branch_map is the time of a live internal node (a coalescence event).
    /// This avoids scanning dead nodes entirely — O(num_samples) regardless of total nodes allocated.
    fn draw_recoalescence(
        &self,
        rng: &mut impl Rng,
        demography: &Demography,
        t_recomb: f64,
    ) -> (Time, usize) {
        let inf = NotNan::new(f64::INFINITY).expect("Not Nan");
        let t_key = NotNan::new(t_recomb).expect("Not Nan");

        // Count coalescences at or below t_recomb to get the lineage count
        let n_coal_below = self.branch_map.range(..=t_key).count();
        let mut k = self.num_samples - n_coal_below;

        // Iterate coalescence events above t_recomb (already sorted by BTreeMap)
        let mut coal_iter = self
            .branch_map
            .range((Excluded(t_key), Excluded(inf)))
            .peekable();

        let mut t_start = t_recomb;

        loop {
            let t_end = coal_iter
                .peek()
                .map_or(f64::INFINITY, |(key, _)| key.into_inner());
            let rate_mult = k as f64;

            // Walk demography epochs within [t_start, t_end), fresh Exp(1) per sub-interval
            let mut current_t = t_start;
            while current_t < t_end {
                let ei = demography.epoch_index_at(current_t);
                let epoch = &demography.epochs[ei];
                let sub_end = t_end.min(demography.epoch_end(ei));
                let e: f64 = Exp::new(1.0).expect("1.0 is valid").sample(rng);
                if let Some(t_coal) = epoch.invert(current_t, e, rate_mult) {
                    if t_coal <= sub_end {
                        let target_idx = Uniform::new(0, k).expect("valid").sample(rng);
                        let target = self.nth_branch_at_time(t_coal, target_idx);
                        return (t_coal, target);
                    }
                }
                current_t = sub_end;
            }

            coal_iter.next();
            k -= 1;
            t_start = t_end;
        }
    }

    /// Sibling of `node` under parent `p`.
    fn sibling(&self, p: NodeID, node: NodeID) -> NodeID {
        let [a, b] = self.children[p].expect("parent must be live internal node");
        if a == node { b } else { a }
    }

    /// Replace child `old` with `new` under parent `p`.
    fn replace_child(&mut self, p: NodeID, old: NodeID, new: NodeID) {
        let ch = self.children[p].as_mut().expect("parent must be live");
        if ch[0] == old {
            ch[0] = new;
        } else {
            ch[1] = new;
        }
    }

    /// SPR: prune `cut_node` from its parent, regraft at `t_coal` on `target_node`'s branch.
    /// Returns (removed_edges, added_edges) using stack-allocated ArrayVecs.
    fn spr(
        &mut self,
        cut_node: usize,
        target_node: usize,
        t_coal: f64,
    ) -> (ArrayVec<Edge, 4>, ArrayVec<Edge, 4>) {
        let c = cut_node;
        let p = self.parent[c].expect("cut_node has parent");
        let s = self.sibling(p, c);
        let g = self.parent[p];
        let t = target_node;

        // Allocate new internal node
        let n = self.time.len();
        self.time.push(t_coal);
        self.parent.push(None);
        self.children.push(None);

        let mut removed = ArrayVec::<Edge, 4>::new();
        removed.push((p, c));
        removed.push((p, s));
        if let Some(g_id) = g {
            removed.push((g_id, p));
        }

        let mut added = ArrayVec::<Edge, 4>::new();

        let g_key = match g {
            Some(g_id) => NotNan::new(self.time[g_id]).expect("Not Nan"),
            None => NotNan::new(f64::INFINITY).expect("Not Nan"),
        };
        let p_key = NotNan::new(self.time[p]).expect("Not Nan");
        let n_key = NotNan::new(t_coal).expect("Not Nan");

        if t == s {
            // Regraft lands on sibling's branch
            if let Some(g_id) = g {
                added.push((g_id, n));
            }
            added.push((n, s));
            added.push((n, c));

            self.children[p] = None;
            self.parent[p] = None;
            self.children[n] = Some([s, c]);
            self.parent[c] = Some(n);
            self.parent[s] = Some(n);
            self.parent[n] = g;
            if let Some(g_id) = g {
                self.replace_child(g_id, p, n);
            }

            Self::bmap_remove(&mut self.branch_map, p_key, c);
            Self::bmap_remove(&mut self.branch_map, p_key, s);
            Self::bmap_remove(&mut self.branch_map, g_key, p);
            Self::bmap_add(&mut self.branch_map, n_key, c);
            Self::bmap_add(&mut self.branch_map, n_key, s);
            Self::bmap_add(&mut self.branch_map, g_key, n);
        } else {
            // General case
            let q = self.parent[t];
            let q_key = match q {
                Some(q_id) => NotNan::new(self.time[q_id]).expect("Not Nan"),
                None => NotNan::new(f64::INFINITY).expect("Not Nan"),
            };

            if let Some(q_id) = q {
                removed.push((q_id, t));
            }

            if let Some(_g_id) = g {
                added.push((_g_id, s));
            }
            if let Some(q_id) = q {
                added.push((q_id, n));
            }
            added.push((n, t));
            added.push((n, c));

            // Prune
            self.children[p] = None;
            self.parent[p] = None;
            self.parent[s] = g;
            if let Some(g_id) = g {
                self.replace_child(g_id, p, s);
            }

            // Regraft
            self.children[n] = Some([t, c]);
            self.parent[c] = Some(n);
            self.parent[t] = Some(n);
            self.parent[n] = q;
            if let Some(q_id) = q {
                self.replace_child(q_id, t, n);
            }

            Self::bmap_remove(&mut self.branch_map, p_key, c);
            Self::bmap_remove(&mut self.branch_map, p_key, s);
            Self::bmap_remove(&mut self.branch_map, g_key, p);
            Self::bmap_remove(&mut self.branch_map, q_key, t);
            Self::bmap_add(&mut self.branch_map, g_key, s);
            Self::bmap_add(&mut self.branch_map, n_key, t);
            Self::bmap_add(&mut self.branch_map, n_key, c);
            Self::bmap_add(&mut self.branch_map, q_key, n);
        }

        for &(par, chi) in &removed {
            self.total_branch_length -= self.time[par] - self.time[chi];
        }
        for &(par, chi) in &added {
            self.total_branch_length += self.time[par] - self.time[chi];
        }

        (removed, added)
    }
}

pub fn sim_ancestry(
    table: &mut TableCollection,
    demography: &Demography,
    num_samples: usize,
    sequence_length: f64,
    recombination_rate: f64,
    seed: u64,
) -> Result<(), TskitError> {
    let mut rng = rand::rngs::Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut x = 0.0;
    let mut tree = CoalTree::draw(&mut rng, num_samples, demography);

    let mut open_edges: HashMap<(usize, usize), f64> = HashMap::new();
    let mut edge_records: Vec<(f64, f64, usize, usize)> = Vec::new();

    for i in 0..tree.time.len() {
        if let Some(p) = tree.parent[i] {
            open_edges.insert((p, i), 0.0);
        }
    }
    while x < sequence_length {
        let rate = recombination_rate * tree.total_branch_length();
        x += Exp::new(rate)
            .expect("Invalid coalescence rate")
            .sample(&mut rng);
        if x >= sequence_length {
            break;
        }
        let (cut_node, t_recomb) = tree.uniform_point(&mut rng);
        let (t_coal, target_node) = tree.draw_recoalescence(&mut rng, demography, t_recomb);
        if target_node == cut_node {
            continue;
        }
        let p = tree.parent[cut_node].expect("cut node must have a parent");
        // If target is p, pruning removes p — its branch becomes the sibling's branch.
        let effective_target = if target_node == p {
            tree.sibling(p, cut_node)
        } else {
            target_node
        };

        let (removed, added) = tree.spr(cut_node, effective_target, t_coal);
        for &(par, chi) in &removed {
            if let Some(left) = open_edges.remove(&(par, chi))
                && x > left
            {
                edge_records.push((left, x, par, chi));
            }
        }
        for &(par, chi) in &added {
            open_edges.insert((par, chi), x);
        }
    }
    for ((par, chi), left) in open_edges.drain() {
        if sequence_length > left {
            edge_records.push((left, sequence_length, par, chi));
        }
    }

    // Write to table collection
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

    table.full_sort(TableSortOptions::default().no_check_integrity())?;
    table.build_index()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::demography::Epoch;
    use proptest::prelude::*;

    fn verify_tree_invariants(
        tree: &CoalTree,
        num_samples: usize,
    ) -> Result<(), proptest::test_runner::TestCaseError> {
        // 1. Exactly one root
        let roots: Vec<usize> = (0..tree.time.len())
            .filter(|&i| tree.parent[i].is_none() && tree.children[i].is_some())
            .collect();
        prop_assert_eq!(roots.len(), 1);

        // 2. Parent↔child consistency + recompute total branch length
        let mut expected_bl = 0.0;
        for i in 0..tree.time.len() {
            if let Some([left, right]) = tree.children[i] {
                prop_assert_eq!(tree.parent[left], Some(i));
                prop_assert_eq!(tree.parent[right], Some(i));
                prop_assert!(tree.time[i] > tree.time[left]);
                prop_assert!(tree.time[i] > tree.time[right]);
            }
            if let Some(p) = tree.parent[i] {
                let ch = tree.children[p].expect("parent is internal");
                prop_assert!(ch[0] == i || ch[1] == i);
                expected_bl += tree.time[p] - tree.time[i];
            }
        }

        // 3. Total branch length
        const EPSILON: f64 = 1e-12;
        prop_assert!((tree.total_branch_length - expected_bl).abs() < EPSILON);

        // 4. Rebuild branch_map from scratch
        let mut expected_map: BTreeMap<NotNan<f64>, Vec<NodeID>> = BTreeMap::new();
        for i in 0..tree.time.len() {
            match tree.parent[i] {
                Some(p_id) => {
                    expected_map
                        .entry(NotNan::new(tree.time[p_id]).expect("finite"))
                        .or_default()
                        .push(i);
                }
                None if tree.children[i].is_some() => {
                    expected_map
                        .entry(NotNan::new(f64::INFINITY).expect("ok"))
                        .or_default()
                        .push(i);
                }
                _ => {}
            }
        }
        prop_assert_eq!(tree.branch_map.len(), expected_map.len());
        for (key, expected_nodes) in &expected_map {
            let mut actual = tree.branch_map.get(key).expect("key present").clone();
            let mut expected = expected_nodes.clone();
            actual.sort();
            expected.sort();
            prop_assert_eq!(actual, expected);
        }

        // 5. All samples reachable from root
        let mut stack = vec![roots[0]];
        let mut leaf_count = 0usize;
        while let Some(node) = stack.pop() {
            match tree.children[node] {
                Some([left, right]) => {
                    stack.push(left);
                    stack.push(right);
                }
                None => leaf_count += 1,
            }
        }
        prop_assert_eq!(leaf_count, num_samples);

        Ok(())
    }

    fn random_demography() -> impl Strategy<Value = Demography> {
        (1usize..=10usize)
            .prop_flat_map(|n| {
                let gaps = proptest::collection::vec(1.0f64..=500.0f64, n - 1);
                let lambdas = proptest::collection::vec(0.1f64..=10.0f64, n);
                let alphas = proptest::collection::vec(-0.02f64..=0.02f64, n);
                (gaps, lambdas, alphas)
            })
            .prop_map(|(gaps, lambdas, mut alphas)| {
                // Last epoch must be constant to guarantee coalescence
                *alphas.last_mut().unwrap() = 0.0;
                let mut times = vec![0.0f64];
                for g in &gaps {
                    times.push(times.last().unwrap() + g);
                }
                Demography {
                    epochs: times
                        .into_iter()
                        .zip(lambdas)
                        .zip(alphas)
                        .map(|((t, lambda), alpha)| Epoch {
                            start_time: t,
                            lambda_start: lambda,
                            alpha,
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
            demography in random_demography(),
        ) {
            let mut rng = rand::rngs::Xoshiro256PlusPlus::seed_from_u64(seed);
            let tree = CoalTree::draw(&mut rng, num_samples, &demography);
            verify_tree_invariants(&tree, num_samples)?;
        }

        #[test]
        fn short_sim_ancestry_does_not_panic(
            seed in 0u64..u64::MAX,
            num_samples in 2usize..=50,
            demography in random_demography(),
        ) {
            let mut table = TableCollection::new(1.0).unwrap();
            sim_ancestry(
                &mut table, &demography, num_samples, 1.0, 1.0, seed
            ).unwrap();
        }

        #[test]
        fn spr_preserves_invariants(
            seed in 0u64..u64::MAX,
            num_samples in 3usize..=30,
            demography in random_demography(),
        ) {
            let mut rng = rand::rngs::Xoshiro256PlusPlus::seed_from_u64(seed);
            let mut tree = CoalTree::draw(&mut rng, num_samples, &demography);

            // Verify the initial tree
            verify_tree_invariants(&tree, num_samples)?;

            // Pick a random branch to cut (uniform_point)
            let (cut_node, t_recomb) = tree.uniform_point(&mut rng);

            // Draw recoalescence
            let (t_coal, target) = tree.draw_recoalescence(&mut rng, &demography, t_recomb);

            // Skip healing events (same node)
            if target == cut_node {
                return Ok(());
            }
            let p = tree.parent[cut_node].expect("cut_node has parent");
            // If target is p, pruning removes p — its branch becomes the sibling's branch.
            let effective_target = if target == p {
                tree.sibling(p, cut_node)
            } else {
                target
            };

            tree.spr(cut_node, effective_target, t_coal);

            // Verify all invariants after SPR
            verify_tree_invariants(&tree, num_samples)?;
        }

        #[test]
        fn spr_chain_preserves_invariants(
            seed in 0u64..u64::MAX,
            num_samples in 3usize..=20,
            n_sprs in 1usize..=20,
            demography in random_demography(),
        ) {
            let mut rng = rand::rngs::Xoshiro256PlusPlus::seed_from_u64(seed);
            let mut tree = CoalTree::draw(&mut rng, num_samples, &demography);

            for _ in 0..n_sprs {
                let (cut_node, t_recomb) = tree.uniform_point(&mut rng);
                let (t_coal, target) = tree.draw_recoalescence(&mut rng, &demography, t_recomb);

                if target == cut_node {
                    continue;
                }
                let p = tree.parent[cut_node].expect("has parent");
                // If target is p, pruning removes p — its branch becomes the sibling's branch.
            let effective_target = if target == p {
                    tree.sibling(p, cut_node)
                } else {
                    target
                };
                tree.spr(cut_node, effective_target, t_coal);
            }

            verify_tree_invariants(&tree, num_samples)?;
        }
    }
}
