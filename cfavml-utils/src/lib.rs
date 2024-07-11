#![doc = include_str!("../README.md")]

pub mod pinning;
mod threadpool;

pub use self::threadpool::{get_or_init_pool, MaybeBorrowedPool};
