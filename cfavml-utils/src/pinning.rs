//! Pin threads to specific CPU cores.
//!
//! This is primarily in order to prevent cache misses causing
//! problem for various operations. In general this step is not required
//! unless you're doing something where maximizing cache efficiency is
//! necessary (like in CFAVML).
//!
//! **NOTE:**
//!
//! On some platforms these operations may be no-ops due to lack of support
//! or missing implementations. `windows`, `linux`, macos`, `freebsd` and `android`
//! are currently supported however.

use std::sync::OnceLock;

use core_affinity::CoreId;

static AVAILABLE_CPUS: OnceLock<Vec<CoreId>> = OnceLock::new();

/// Pin the current thread to the target CPU specified by the `cpu_id`.
///
/// **NOTE:**
///
/// The provided number is an _index_ not the actual CPU ID, the system will
/// automatically select the right _relative_ CPU id based on the index.
///
/// On some platforms these operations may be no-ops due to lack of support
/// or missing implementations. `windows`, `linux`, macos`, `freebsd` and `android`
/// are currently supported however.
pub fn pin_current(cpu_id_idx: usize) -> bool {
    let available =
        AVAILABLE_CPUS.get_or_init(|| core_affinity::get_core_ids().unwrap_or_default());
    let num_available = available.len();

    if num_available == 0 {
        return false;
    }

    if num_available <= cpu_id_idx {
        if cfg!(debug_assertions) {
            panic!(
                "Cannot pin to CPU that does not exist {num_available} available CPUs, \
                {cpu_id_idx} provided index"
            );
        }
        return false;
    }

    let cpu_id = available[cpu_id_idx];
    core_affinity::set_for_current(cpu_id)
}
