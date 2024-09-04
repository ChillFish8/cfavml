mod arithmetic_ops;
mod broadcast_shim;
mod broadcastable_op;
mod utils;

/// Standard 'Fast' operation types.
///
/// These are exposed via a custom trait due to orphan rules, typically you want to use
/// the function routines instead of the trait based routines for convenience.
pub mod ops {
    pub use super::arithmetic_ops::{
        add,
        div,
        mul,
        sub,
        AddFast,
        DivFast,
        MulFast,
        SubFast,
    };
}

/// A set of common imports you want imported when use trait based approaches.
pub mod prelude {
    pub use super::ops::*;
}
