use pyo3::prelude::*;

/// Demography model.
mod demography;

/// Implements SMC' simulations
mod simulations;

/// A Python module implemented in Rust.
#[pymodule]
mod smc_prime {
    use pyo3::exceptions::PyRuntimeError;
    use pyo3::prelude::*;

    use crate::demography;
    use crate::simulations;

    /// Parse an msprime.Demography object into our Demography via attribute access.
    fn parse_msprime_demography(demo: &Bound<'_, PyAny>) -> PyResult<demography::Demography> {
        let pops = demo.getattr("populations")?;
        let num_pops: usize = pops.len()?;
        if num_pops != 1 {
            return Err(PyRuntimeError::new_err(format!(
                "smc_prime only supports single-population models, got {num_pops} populations"
            )));
        }

        // Check migration matrix for non-zero entries
        let mig_matrix = demo.getattr("migration_matrix")?;
        let flat: Vec<f64> = mig_matrix
            .call_method0("flatten")?
            .call_method0("tolist")?
            .extract()?;
        if flat.iter().any(|&x| x != 0.0) {
            return Err(PyRuntimeError::new_err(
                "smc_prime does not support migration; all migration rates must be zero",
            ));
        }

        // Extract initial population parameters
        let pop0 = pops.get_item(0)?;
        let init_size: f64 = pop0.getattr("initial_size")?.extract()?;
        let init_growth: f64 = pop0.getattr("growth_rate")?.extract()?;

        let mut epoch_tuples: Vec<(f64, f64, f64)> = vec![(0.0, init_size, init_growth)];

        // Collect PopulationParametersChange events
        let events = demo.getattr("events")?;
        for event in events.try_iter()? {
            let event: Bound<'_, PyAny> = event?;
            let type_name: String = event.get_type().getattr("__name__")?.extract()?;
            if type_name == "PopulationParametersChange" {
                let time: f64 = event.getattr("time")?.extract()?;
                let size: f64 = event.getattr("initial_size")?.extract()?;
                let growth: f64 = event.getattr("growth_rate")?.extract().unwrap_or(0.0);
                epoch_tuples.push((time, size, growth));
            }
        }

        epoch_tuples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        demography::Demography::piecewise_exponential_epochs(&epoch_tuples)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Run SMC' algorithm
    #[pyfunction]
    #[pyo3(signature = (population_size, num_samples=2, sequence_length=1.0, recombination_rate=1.0, random_seed=None))]
    pub fn sim_ancestry(
        py: Python<'_>,
        population_size: &Bound<'_, PyAny>,
        num_samples: usize,
        sequence_length: Option<f64>,
        recombination_rate: Option<f64>,
        random_seed: Option<u64>,
    ) -> PyResult<Py<PyAny>> {
        let random_seed = random_seed.unwrap_or_else(rand::random);
        if num_samples < 2 {
            return Err(PyRuntimeError::new_err("num_samples must be at least 2"));
        }
        let demography = if let Ok(ne) = population_size.extract::<f64>() {
            demography::Demography::constant(ne)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        } else if let Ok(triples) = population_size.extract::<Vec<(f64, f64, f64)>>() {
            demography::Demography::piecewise_exponential_epochs(&triples)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        } else if let Ok(pairs) = population_size.extract::<Vec<(f64, f64)>>() {
            demography::Demography::piecewise_constant_epochs(&pairs)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        } else if population_size.getattr("populations").is_ok() {
            parse_msprime_demography(population_size)?
        } else {
            return Err(PyRuntimeError::new_err(
                "population_size must be a number, a list of (time, size[, growth_rate]) tuples, \
                 or an msprime.Demography object",
            ));
        };

        let sequence_length = sequence_length.unwrap_or(1.0);
        let recombination_rate = recombination_rate.unwrap_or(0.0);

        let mut shared = tskit2tskit::SharedTableCollection::new(py, sequence_length)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        py.detach(|| -> Result<(), PyErr> {
            // SAFETY: safe if tskit-rust and tskit-python share the same
            // tsk_table_collection_t layout (same tskit-c version).
            Ok(unsafe {
                shared.with_mut_tables(|tables| -> Result<(), tskit2tskit::Error> {
                    simulations::sim_ancestry(
                        tables,
                        &demography,
                        num_samples,
                        sequence_length,
                        recombination_rate,
                        random_seed,
                    )?;
                    Ok(())
                })
            }?)
        })?;

        Ok(shared
            .into_python_tree_sequence(py)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?)
    }
}
