//! A CFAVML configured rayon threadpool.
//!
//! This is a threadpool which is typically global, configured by the `CFAVML_*` env vars
//! and specialized for high performance compute.
//!
//! The main difference for wrapping the [rayon::ThreadPool] is threads are pinned to
//! physical CPU cores to prevent scheduling issues poisoning the cache.
//!
//! ```
//! use cfavml_utils::get_or_init_pool;
//!
//! let pool = get_or_init_pool();
//! pool.spawn(move || {
//!     println!("Hello!");
//! });
//! ```
//!

use std::ops::Deref;
use std::sync::OnceLock;

/// Gets or initializes the global CFAVML thread pool.
pub fn get_or_init_pool() -> MaybeBorrowedPool {
    static SHARED_THREADPOOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();

    let global_pool = SHARED_THREADPOOL
        .get_or_init(|| {
            let no_cache = config_bool("CFAVML_NO_CACHE_THREADPOOL");
            if no_cache {
                None
            } else {
                Some(create_pool())
            }
        })
        .as_ref();

    match global_pool {
        None => MaybeBorrowedPool::Owned(create_pool()),
        Some(pool) => MaybeBorrowedPool::Borrowed(pool),
    }
}

/// A borrowed or owned threadpool.
///
/// If the system is using the global pool, it will return a borrowed
/// value, otherwise it will return a newly initialised pool.
pub enum MaybeBorrowedPool {
    Borrowed(&'static rayon::ThreadPool),
    Owned(rayon::ThreadPool),
}

impl Deref for MaybeBorrowedPool {
    type Target = rayon::ThreadPool;

    fn deref(&self) -> &Self::Target {
        match self {
            MaybeBorrowedPool::Borrowed(pool) => pool,
            MaybeBorrowedPool::Owned(pool) => pool,
        }
    }
}

fn create_pool() -> rayon::ThreadPool {
    let num_threads = std::cmp::min(config_num_threads(), num_cpus::get_physical());

    let no_pinning = config_bool("CFAVML_NO_PINNING");
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .start_handler(move |thread_id| {
            if !no_pinning {
                crate::pinning::pin_current(thread_id);
            }
        })
        .build()
        .expect("Build rayon threadpool")
}

fn config_num_threads() -> usize {
    if let Ok(value) = std::env::var("CFAVML_NUM_THREADS") {
        return value.parse().unwrap_or_else(|_| {
            load_debug(format!(
                "`CFAVML_NUM_THREADS` env var was provided but has \
                    invalid data {value:?}, using default value"
            ));
            num_cpus::get_physical()
        });
    }

    #[cfg(feature = "env-var-compat")]
    if let Ok(value) = std::env::var("OMP_NUM_THREADS") {
        return value.parse().unwrap_or_else(|_| {
            load_debug(format!(
                "`OMP_NUM_THREADS` env var was provided but has \
                    invalid data {value:?}, using default value"
            ));
            num_cpus::get_physical()
        });
    }

    #[cfg(feature = "env-var-compat")]
    if let Ok(value) = std::env::var("OPENBLAS_NUM_THREADS") {
        return value.parse().unwrap_or_else(|_| {
            load_debug(format!(
                "`OMP_NUM_THREADS` env var was provided but has \
                    invalid data {value:?}, using default value"
            ));
            num_cpus::get_physical()
        });
    }

    num_cpus::get_physical()
}

fn config_bool(env: &str) -> bool {
    std::env::var(env)
        .map(|v| cast_bool(&v))
        .unwrap_or_default()
}

fn load_debug(msg: String) {
    static SHOULD_LOG: OnceLock<bool> = OnceLock::new();
    let should_log = *SHOULD_LOG.get_or_init(|| {
        std::env::var("CFAVML_DEBUG")
            .map(|v| cast_bool(&v))
            .unwrap_or_default()
    });

    if should_log {
        eprintln!("CFAVML_DEBUG: {msg}");
    }
}

fn cast_bool(value: &str) -> bool {
    static TRUE_VALUES: &[&str] = &["1", "true", "TRUE"];
    TRUE_VALUES.contains(&value)
}
