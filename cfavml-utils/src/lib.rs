#![doc = include_str!("../README.md")]

pub mod aligned_buffer;
pub mod pinning;
mod threadpool;

pub use self::threadpool::{get_or_init_pool, MaybeBorrowedPool};
