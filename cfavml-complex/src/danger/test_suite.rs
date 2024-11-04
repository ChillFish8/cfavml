macro_rules! test_suite {
    ($t:ident, $im:ident) => {
        paste::paste! {
            #[test]
            fn [<test_ $im:lower _ $t _suite>]() {
                unsafe { crate::danger::impl_test::test_suite_impl::<$t, $im>() }
            }
        }
    };
}

#[cfg(all(target_feature = "avx2", test))]
mod avx2_tests {
    use super::*;
    test_suite!(f32, Avx2Complex);
    test_suite!(f64, Avx2Complex);
}
#[cfg(all(target_feature = "avx2", target_feature = "fma", test))]
mod avx2fma_tests {
    use super::*;

    test_suite!(f32, Avx2ComplexFma);
    test_suite!(f64, Avx2ComplexFma);
}
