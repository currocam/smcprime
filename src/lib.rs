use pyo3::prelude::*;
/// Implements pointer conversions between Rust and Python tskit objects
mod ffi;
/// Implements SMC++ simulations
mod simulations;

/// A Python module implemented in Rust.
#[pymodule]
mod smc_prime {
    use enterpolation::linear::Linear;
    use pyo3::exceptions::PyRuntimeError;
    use pyo3::prelude::*;

    use crate::ffi;
    use crate::simulations;

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

    /// Run SMC' algorithm with variable Ne
    #[pyfunction]
    #[pyo3(signature = (population_size_fn, num_samples=2, sequence_length=1.0, recombination_rate=1.0, random_seed=None, granularity=None))]
    pub fn sim_ancestry_variable(
        py: Python<'_>,
        population_size_fn: &Bound<'_, PyAny>,
        num_samples: usize,
        sequence_length: Option<f64>,
        recombination_rate: Option<f64>,
        random_seed: Option<u64>,
        granularity: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        if !population_size_fn.is_callable() {
            return Err(PyRuntimeError::new_err(
                "population_size_fn must be a callable taking one position argument",
            ));
        }
        if num_samples < 2 {
            return Err(PyRuntimeError::new_err("num_samples must be at least 2"));
        }

        let sequence_length = sequence_length.unwrap_or(1.0);
        if !sequence_length.is_finite() || sequence_length <= 0.0 {
            return Err(PyRuntimeError::new_err(
                "sequence_length must be a positive finite float",
            ));
        }
        let granularity = granularity.unwrap_or(1000).max(2);
        let step = sequence_length / (granularity - 1) as f64;

        let mut xs = Vec::with_capacity(granularity);
        let mut nes = Vec::with_capacity(granularity);
        for i in 0..granularity {
            let x = i as f64 * step;
            let ne = population_size_fn
                .call1((x,))?
                .extract::<f64>()
                .map_err(|_| {
                    PyRuntimeError::new_err(
                        "population_size_fn(position) must return a positive finite float",
                    )
                })?;
            if !ne.is_finite() || ne <= 0.0 {
                return Err(PyRuntimeError::new_err(
                    "population_size_fn(position) must return a positive finite float",
                ));
            }
            xs.push(x);
            nes.push(ne);
        }

        let lin = Linear::builder()
            .elements(nes)
            .knots(xs)
            .build()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let tables = simulations::sim_ancestry_variable(
            lin,
            num_samples,
            sequence_length,
            recombination_rate.unwrap_or(0.0),
            random_seed.unwrap_or_else(rand::random),
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        ffi::table_collection_into_python_tree_sequence(py, tables)
    }
}
