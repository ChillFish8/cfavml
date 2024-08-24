mod add_op;
mod broadcast_shim;
mod utils;
mod broadcastable_op;


/// Standard 'Fast' operation types.
/// 
/// These are exposed via a custom trait due to orphan rules, typically you want to use
/// the function routines instead of the trait based routines for convenience.
pub mod ops {
    pub use super::add_op::AddFast;
}

/// A set of common imports you want imported when use trait based approaches.
pub mod prelude {
    pub use super::ops::*;
}