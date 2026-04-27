use std::time::Instant;
use tskit::TableCollection;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let ne: f64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10_000.0);
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
    let seq_len: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10.0);
    let r: f64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let n_reps: u64 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(1);

    eprintln!("Ne={ne}, n={n}, L={seq_len}, r={r}, reps={n_reps}");

    let demography = smc_prime::demography::Demography::constant(ne).unwrap();

    let t0 = Instant::now();
    for seed in 1..=n_reps {
        let mut table = TableCollection::new(seq_len).unwrap();
        smc_prime::simulations::sim_ancestry(&mut table, &demography, n, seq_len, r, seed).unwrap();
    }
    let elapsed = t0.elapsed();

    eprintln!(
        "{n_reps} reps in {:.2}s ({:.2} ms/rep)",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() * 1000.0 / n_reps as f64
    );
}
