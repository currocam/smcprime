use pyo3::prelude::*;
/// Implements pointer conversions between Rust and Python tskit objects
mod ffi;
/// Implements SMC++ simulations
mod simulations;

/// A Python module implemented in Rust.
#[pymodule]
mod smc_prime {
    use pyo3::exceptions::PyRuntimeError;
    use pyo3::prelude::*;

    use crate::ffi;
    use crate::simulations;

    /// Run a haploid Wright-Fisher simulation
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
            simulations::Demography::constant(ne)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        } else if let Ok(pairs) = population_size.extract::<Vec<(f64, f64)>>() {
            simulations::Demography::from_tuples(&pairs)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        } else {
            return Err(PyRuntimeError::new_err(
                "population_size must be a number or a list of (time, size) tuples",
            ));
        };
        let tables = simulations::sim_ancestry(
            &demography,
            num_samples,
            sequence_length.unwrap_or(1.0),
            recombination_rate.unwrap_or(0.0),
            random_seed,
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        ffi::table_collection_into_python_tree_sequence(py, tables)
    }
}
