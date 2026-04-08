// Tools to convert Rust TableCollection to Python tskit objects

use pyo3::prelude::*;
use tskit::TableCollection;

/// # Safety
/// `py_obj` must be a valid pointer to a `_tskit.TableCollection` Python object,
/// whose layout has `tsk_table_collection_t` immediately after `PyObject_HEAD`.
pub unsafe fn read_tsk_ptr(
    py_obj: *mut pyo3::ffi::PyObject,
) -> *mut tskit::bindings::tsk_table_collection_t {
    unsafe {
        // _tskit.TableCollection stores a *pointer* to a heap-allocated
        // tsk_table_collection_t immediately after PyObject_HEAD (offset 16).
        let offset = std::mem::size_of::<pyo3::ffi::PyObject>();
        let ptr_field =
            (py_obj as *mut u8).add(offset) as *mut *mut tskit::bindings::tsk_table_collection_t;
        *ptr_field
    }
}

/// Converts a Rust `TableCollection` into a Python `tskit.TreeSequence`.
pub fn table_collection_into_python_tree_sequence(
    py: Python<'_>,
    rust_tables: TableCollection,
) -> PyResult<Py<PyAny>> {
    let sequence_length: f64 = rust_tables.sequence_length().into();

    // Create an empty _tskit.TableCollection
    let ll_tskit = py.import("_tskit")?;
    let py_ll_tc = ll_tskit
        .getattr("TableCollection")?
        .call1((sequence_length,))?;

    // Copy Rust table data into Python's tsk_table_collection_t
    unsafe {
        let py_obj_ptr = py_ll_tc.as_ptr();
        let dest_ptr = read_tsk_ptr(py_obj_ptr);
        tskit::bindings::tsk_table_collection_free(dest_ptr);
        let rv = tskit::bindings::tsk_table_collection_copy(rust_tables.as_ptr(), dest_ptr, 0);
        if rv < 0 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "tsk_table_collection_copy failed with code {rv}"
            )));
        }
    }
    // Wrap in high-level tskit.TableCollection and create tree sequence
    let tskit_mod = py.import("tskit")?;
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("ll_tables", &py_ll_tc)?;
    let py_tc = tskit_mod
        .getattr("TableCollection")?
        .call((), Some(&kwargs))?;
    py_tc.call_method0("sort")?;
    let ts = py_tc.call_method0("tree_sequence")?;
    Ok(ts.unbind())
}
