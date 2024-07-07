use std::fmt::Debug;
use std::iter::zip;

use rand::distributions::{Distribution, Standard};

use crate::danger::SimdRegister;
use crate::math::{AutoMath, Math};
use crate::test_utils::get_sample_vectors;

/// Runs a set of generic test suites to ensure a given impl is working correctly.
pub(crate) unsafe fn test_suite_impl<T, R>()
where
    T: Copy + Debug + PartialEq,
    R: SimdRegister<T>,
    AutoMath: Math<T>,
    Standard: Distribution<T>,
{
    let (small_sample_l1, small_sample_l2) =
        get_sample_vectors::<T>(R::elements_per_lane());
    let (large_sample_l1, large_sample_l2) =
        get_sample_vectors::<T>(R::elements_per_dense());

    // Single lane handling.
    {
        let l1 = R::load(small_sample_l1.as_ptr());
        let l2 = R::load(small_sample_l2.as_ptr());
        let res = R::add(l1, l2);
        let value = R::sum_to_value(res);
        let expected_value = zip(small_sample_l1.iter(), small_sample_l2.iter())
            .map(|(a, b)| AutoMath::add(*a, *b))
            .fold(AutoMath::zero(), |a, b| AutoMath::add(a, b));
        assert!(
            AutoMath::is_close(value, expected_value),
            "Addition and sum test failed on single task"
        );
    }

    {
        let l1 = R::load(small_sample_l1.as_ptr());
        let l2 = R::load(small_sample_l2.as_ptr());
        let res = R::sub(l1, l2);
        let value = R::sum_to_value(res);
        let expected_value = zip(small_sample_l1.iter(), small_sample_l2.iter())
            .map(|(a, b)| AutoMath::sub(*a, *b))
            .fold(AutoMath::zero(), |a, b| AutoMath::add(a, b));
        assert!(
            AutoMath::is_close(value, expected_value),
            "Subtraction and sum test failed on single task"
        );
    }

    {
        let l1 = R::load(small_sample_l1.as_ptr());
        let l2 = R::load(small_sample_l2.as_ptr());
        let res = R::mul(l1, l2);
        let value = R::sum_to_value(res);
        let expected_value = zip(small_sample_l1.iter(), small_sample_l2.iter())
            .map(|(a, b)| AutoMath::mul(*a, *b))
            .fold(AutoMath::zero(), |a, b| AutoMath::add(a, b));
        assert!(
            AutoMath::is_close(value, expected_value),
            "Multiplication and sum test failed on single task"
        );
    }

    {
        let l1 = R::load(small_sample_l1.as_ptr());
        let l2 = R::load(small_sample_l2.as_ptr());
        let l3 = R::filled(AutoMath::zero());
        let res = R::fmadd(l1, l2, l3);
        let value = R::sum_to_value(res);
        let expected_value = zip(small_sample_l1.iter(), small_sample_l2.iter())
            .map(|(a, b)| AutoMath::mul(*a, *b))
            .fold(AutoMath::zero(), |a, b| AutoMath::add(a, b));
        assert!(
            AutoMath::is_close(value, expected_value),
            "Fmadd and sum test failed on single task"
        );
    }

    {
        let l1 = R::load(small_sample_l1.as_ptr());
        let l2 = R::load(small_sample_l2.as_ptr());
        let res = R::div(l1, l2);
        let value = R::sum_to_value(res);
        let expected_value = zip(small_sample_l1.iter(), small_sample_l2.iter())
            .map(|(a, b)| AutoMath::div(*a, *b))
            .fold(AutoMath::zero(), |a, b| AutoMath::add(a, b));
        assert!(
            AutoMath::is_close(value, expected_value),
            "Division and sum test failed on single task"
        );
    }

    {
        let l1 = R::load(small_sample_l1.as_ptr());
        let l2 = R::load(small_sample_l2.as_ptr());
        let res = R::max(l1, l2);
        let value = R::max_to_value(res);
        let expected_value = zip(small_sample_l1.iter(), small_sample_l2.iter())
            .map(|(a, b)| AutoMath::cmp_max(*a, *b))
            .fold(AutoMath::min(), |a, b| AutoMath::cmp_max(a, b));
        assert_eq!(
            value, expected_value,
            "Max and max_to_value test failed on single task"
        );
    }

    {
        let l1 = R::load(small_sample_l1.as_ptr());
        let l2 = R::load(small_sample_l2.as_ptr());
        let res = R::min(l1, l2);
        let value = R::min_to_value(res);
        let expected_value = zip(small_sample_l1.iter(), small_sample_l2.iter())
            .map(|(a, b)| AutoMath::cmp_min(*a, *b))
            .fold(AutoMath::max(), |a, b| AutoMath::cmp_min(a, b));
        assert_eq!(
            value, expected_value,
            "min and min_to_value test failed on single task"
        );
    }

    {
        let l1 = R::load(small_sample_l1.as_ptr());
        let l2 = R::load(small_sample_l2.as_ptr());
        let res = R::add(l1, l2);

        let mut target_output = vec![AutoMath::zero(); R::elements_per_lane()];
        R::write(target_output.as_mut_ptr(), res);

        let expected_output = zip(small_sample_l1.iter(), small_sample_l2.iter())
            .map(|(a, b)| AutoMath::add(*a, *b))
            .collect::<Vec<_>>();

        assert_eq!(
            target_output, expected_output,
            "Single lane write failed task"
        );
    }

    // Dense lane handling.
    {
        let l1 = R::load_dense(large_sample_l1.as_ptr());
        let l2 = R::load_dense(large_sample_l2.as_ptr());
        let res = R::add_dense(l1, l2);
        let reg = R::sum_to_register(res);
        let value = R::sum_to_value(reg);

        let expected_value = zip(large_sample_l1.iter(), large_sample_l2.iter())
            .map(|(a, b)| AutoMath::add(*a, *b))
            .fold(AutoMath::zero(), |a, b| AutoMath::add(a, b));

        assert!(
            AutoMath::is_close(value, expected_value),
            "Addition and sum test failed on dense task"
        );
    }

    {
        let l1 = R::load_dense(large_sample_l1.as_ptr());
        let l2 = R::load_dense(large_sample_l2.as_ptr());
        let res = R::sub_dense(l1, l2);
        let reg = R::sum_to_register(res);
        let value = R::sum_to_value(reg);

        let expected_value = zip(large_sample_l1.iter(), large_sample_l2.iter())
            .map(|(a, b)| AutoMath::sub(*a, *b))
            .fold(AutoMath::zero(), |a, b| AutoMath::add(a, b));

        assert!(
            AutoMath::is_close(value, expected_value),
            "Subtraction and sum test failed on dense task"
        );
    }

    {
        let l1 = R::load_dense(large_sample_l1.as_ptr());
        let l2 = R::load_dense(large_sample_l2.as_ptr());
        let res = R::mul_dense(l1, l2);
        let reg = R::sum_to_register(res);
        let value = R::sum_to_value(reg);

        let expected_value = zip(large_sample_l1.iter(), large_sample_l2.iter())
            .map(|(a, b)| AutoMath::mul(*a, *b))
            .fold(AutoMath::zero(), |a, b| AutoMath::add(a, b));

        assert!(
            AutoMath::is_close(value, expected_value),
            "Multiplication and sum test failed on dense task"
        );
    }

    {
        let l1 = R::load_dense(large_sample_l1.as_ptr());
        let l2 = R::load_dense(large_sample_l2.as_ptr());
        let l3 = R::filled_dense(AutoMath::zero());
        let res = R::fmadd_dense(l1, l2, l3);
        let reg = R::sum_to_register(res);
        let value = R::sum_to_value(reg);

        let expected_value = zip(large_sample_l1.iter(), large_sample_l2.iter())
            .map(|(a, b)| AutoMath::mul(*a, *b))
            .fold(AutoMath::zero(), |a, b| AutoMath::add(a, b));

        assert!(
            AutoMath::is_close(value, expected_value),
            "Fmadd and sum test failed on dense task"
        );
    }

    {
        let l1 = R::load_dense(large_sample_l1.as_ptr());
        let l2 = R::load_dense(large_sample_l2.as_ptr());
        let res = R::div_dense(l1, l2);
        let reg = R::sum_to_register(res);
        let value = R::sum_to_value(reg);

        let expected_value = zip(large_sample_l1.iter(), large_sample_l2.iter())
            .map(|(a, b)| AutoMath::div(*a, *b))
            .fold(AutoMath::zero(), |a, b| AutoMath::add(a, b));

        assert!(
            AutoMath::is_close(value, expected_value),
            "Division and sum test failed on dense task"
        );
    }

    {
        let l1 = R::load_dense(large_sample_l1.as_ptr());
        let l2 = R::load_dense(large_sample_l2.as_ptr());
        let res = R::max_dense(l1, l2);
        let reg = R::max_to_register(res);
        let value = R::max_to_value(reg);

        let expected_value = zip(large_sample_l1.iter(), large_sample_l2.iter())
            .map(|(a, b)| AutoMath::cmp_max(*a, *b))
            .fold(AutoMath::min(), |a, b| AutoMath::cmp_max(a, b));

        assert_eq!(
            value, expected_value,
            "Max and max_to_value test failed on dense task"
        );
    }

    {
        let l1 = R::load_dense(large_sample_l1.as_ptr());
        let l2 = R::load_dense(large_sample_l2.as_ptr());
        let res = R::min_dense(l1, l2);
        let reg = R::min_to_register(res);
        let value = R::min_to_value(reg);

        let expected_value = zip(large_sample_l1.iter(), large_sample_l2.iter())
            .map(|(a, b)| AutoMath::cmp_min(*a, *b))
            .fold(AutoMath::max(), |a, b| AutoMath::cmp_min(a, b));

        assert_eq!(
            value, expected_value,
            "min and min_to_value test failed on dense task"
        );
    }

    {
        let l1 = R::load_dense(large_sample_l1.as_ptr());
        let l2 = R::load_dense(large_sample_l2.as_ptr());
        let res = R::add_dense(l1, l2);

        let mut target_output = vec![AutoMath::zero(); R::elements_per_dense()];
        R::write_dense(target_output.as_mut_ptr(), res);

        let expected_output = zip(large_sample_l1.iter(), large_sample_l2.iter())
            .map(|(a, b)| AutoMath::add(*a, *b))
            .collect::<Vec<_>>();

        assert_eq!(
            target_output, expected_output,
            "Dense lane write failed dense task"
        );
    }
}
