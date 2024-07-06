use crate::danger::SimdRegister;

/// Runs a set of generic test suites to ensure a given impl is working correctly.
pub(crate) unsafe fn test_suite_impl_f32<R>()
where
    R: SimdRegister<f32>,
{
    let small_sample_l1 = vec![1.0; R::elements_per_lane()];
    let small_sample_l2 = vec![2.0; R::elements_per_lane()];

    let large_sample_l1 = vec![1.0; R::elements_per_dense()];
    let large_sample_l2 = vec![2.0; R::elements_per_dense()];

    // Single lane handling.
    {
        let l1 = R::load(small_sample_l1.as_ptr());
        let l2 = R::load(small_sample_l2.as_ptr());
        let res = R::add(l1, l2);
        let value = R::sum_to_value(res);
        let expected_value = (1.0 * R::elements_per_lane() as f32)
            + (2.0 * R::elements_per_lane() as f32);
        assert_eq!(
            value, expected_value,
            "Addition and sum test failed on single register f32 task"
        );
    }

    {
        let l1 = R::load(small_sample_l1.as_ptr());
        let l2 = R::load(small_sample_l2.as_ptr());
        let res = R::sub(l1, l2);
        let value = R::sum_to_value(res);
        let expected_value = -1.0 * R::elements_per_lane() as f32;
        assert_eq!(
            value, expected_value,
            "Subtraction and sum test failed on single register f32 task"
        );
    }

    {
        let l1 = R::load(small_sample_l1.as_ptr());
        let l2 = R::load(small_sample_l2.as_ptr());
        let res = R::mul(l1, l2);
        let value = R::sum_to_value(res);
        let expected_value = 2.0 * R::elements_per_lane() as f32;
        assert_eq!(
            value, expected_value,
            "Multiplication and sum test failed on single register f32 task"
        );
    }

    {
        let l1 = R::load(small_sample_l1.as_ptr());
        let l2 = R::load(small_sample_l2.as_ptr());
        let l3 = R::filled(2.0);
        let res = R::fmadd(l1, l2, l3);
        let value = R::sum_to_value(res);
        let expected_value = 4.0 * R::elements_per_lane() as f32;
        assert_eq!(
            value, expected_value,
            "Fmadd and sum test failed on single register f32 task"
        );
    }

    {
        let l1 = R::load(small_sample_l1.as_ptr());
        let l2 = R::load(small_sample_l2.as_ptr());
        let res = R::div(l1, l2);
        let value = R::sum_to_value(res);
        let expected_value = 0.5 * R::elements_per_lane() as f32;
        assert_eq!(
            value, expected_value,
            "Division and sum test failed on single register f32 task"
        );
    }

    {
        let l1 = R::load(small_sample_l1.as_ptr());
        let l2 = R::load(small_sample_l2.as_ptr());
        let res = R::max(l1, l2);
        let value = R::max_to_value(res);
        assert_eq!(
            value, 2.0,
            "Max and max_to_value test failed on single register f32 task"
        );
    }

    {
        let l1 = R::load(small_sample_l1.as_ptr());
        let l2 = R::load(small_sample_l2.as_ptr());
        let res = R::min(l1, l2);
        let value = R::min_to_value(res);
        assert_eq!(
            value, 1.0,
            "min and min_to_value test failed on single register f32 task"
        );
    }

    {
        let l1 = R::load(small_sample_l1.as_ptr());
        let l2 = R::load(small_sample_l2.as_ptr());
        let res = R::add(l1, l2);

        let mut target_output = vec![0.0; R::elements_per_lane()];
        R::write(target_output.as_mut_ptr(), res);
        assert_eq!(
            target_output,
            vec![3.0; R::elements_per_lane()],
            "Single lane write failed f32 task"
        );
    }

    // Dense lane handling.
    {
        let l1 = R::load_dense(large_sample_l1.as_ptr());
        let l2 = R::load_dense(large_sample_l2.as_ptr());
        let res = R::add_dense(l1, l2);
        let reg = R::sum_to_register(res);
        let value = R::sum_to_value(reg);
        let expected_value = (1.0 * R::elements_per_dense() as f32)
            + (2.0 * R::elements_per_dense() as f32);
        assert_eq!(
            value, expected_value,
            "Addition and sum test failed on dense register f32 task"
        );
    }

    {
        let l1 = R::load_dense(large_sample_l1.as_ptr());
        let l2 = R::load_dense(large_sample_l2.as_ptr());
        let res = R::sub_dense(l1, l2);
        let reg = R::sum_to_register(res);
        let value = R::sum_to_value(reg);
        let expected_value = -1.0 * R::elements_per_dense() as f32;
        assert_eq!(
            value, expected_value,
            "Subtraction and sum test failed on dense register f32 task"
        );
    }

    {
        let l1 = R::load_dense(large_sample_l1.as_ptr());
        let l2 = R::load_dense(large_sample_l2.as_ptr());
        let res = R::mul_dense(l1, l2);
        let reg = R::sum_to_register(res);
        let value = R::sum_to_value(reg);
        let expected_value = 2.0 * R::elements_per_dense() as f32;
        assert_eq!(
            value, expected_value,
            "Multiplication and sum test failed on dense register f32 task"
        );
    }

    {
        let l1 = R::load_dense(large_sample_l1.as_ptr());
        let l2 = R::load_dense(large_sample_l2.as_ptr());
        let l3 = R::filled_dense(2.0);
        let res = R::fmadd_dense(l1, l2, l3);
        let reg = R::sum_to_register(res);
        let value = R::sum_to_value(reg);
        let expected_value = 4.0 * R::elements_per_dense() as f32;
        assert_eq!(
            value, expected_value,
            "Fmadd and sum test failed on dense register f32 task"
        );
    }

    {
        let l1 = R::load_dense(large_sample_l1.as_ptr());
        let l2 = R::load_dense(large_sample_l2.as_ptr());
        let res = R::div_dense(l1, l2);
        let reg = R::sum_to_register(res);
        let value = R::sum_to_value(reg);
        let expected_value = 0.5 * R::elements_per_dense() as f32;
        assert_eq!(
            value, expected_value,
            "Division and sum test failed on dense register f32 task"
        );
    }

    {
        let l1 = R::load_dense(large_sample_l1.as_ptr());
        let l2 = R::load_dense(large_sample_l2.as_ptr());
        let res = R::max_dense(l1, l2);
        let reg = R::max_to_register(res);
        let value = R::max_to_value(reg);
        assert_eq!(
            value, 2.0,
            "Max and max_to_value test failed on dense register f32 task"
        );
    }

    {
        let l1 = R::load_dense(large_sample_l1.as_ptr());
        let l2 = R::load_dense(large_sample_l2.as_ptr());
        let res = R::min_dense(l1, l2);
        let reg = R::max_to_register(res);
        let value = R::min_to_value(reg);
        assert_eq!(
            value, 1.0,
            "min and min_to_value test failed on dense register f32 task"
        );
    }

    {
        let l1 = R::load_dense(large_sample_l1.as_ptr());
        let l2 = R::load_dense(large_sample_l2.as_ptr());
        let res = R::add_dense(l1, l2);

        let mut target_output = vec![0.0; R::elements_per_dense()];
        R::write_dense(target_output.as_mut_ptr(), res);
        assert_eq!(
            target_output,
            vec![3.0; R::elements_per_dense()],
            "Dense lane write failed f32 task"
        );
    }
}
