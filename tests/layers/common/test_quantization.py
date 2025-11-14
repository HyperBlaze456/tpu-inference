import jax.numpy as jnp

from tpu_inference.layers.common.quantization import quantize_kv


def test_quantize_kv_float8_e4m3fn():
    """Tests the quantize_kv function with float8_e4m3fn dtype."""
    key = jnp.array([-1.0, 0.5, 1.0, 1.5])
    value = jnp.array([2.0, 0.0, -2.0, -3.0])
    kv_cache_quantized_dtype = jnp.float8_e4m3fn
    k_scale = 0.1
    v_scale = 0.2

    quantized_key, quantized_value = quantize_kv(key, value,
                                                 kv_cache_quantized_dtype,
                                                 k_scale, v_scale)

    # Expected key: key / k_scale -> clip -> astype
    # [-10., 5., 10., 15.] are within float8_e4m3fn range
    expected_key = jnp.array([-10.0, 5.0, 10.0, 15.0], dtype=jnp.float8_e4m3fn)

    # Expected value: value / v_scale -> clip -> astype
    # [10., 0., -10., -15.] are within float8_e4m3fn range
    expected_value = jnp.array([10.0, 0.0, -10.0, -15.0],
                               dtype=jnp.float8_e4m3fn)

    assert jnp.array_equal(quantized_key, expected_key)
    assert jnp.array_equal(quantized_value, expected_value)

    # Test clipping
    dtype_info = jnp.finfo(kv_cache_quantized_dtype)
    minval, maxval = float(dtype_info.min), float(dtype_info.max)

    # Values that will be outside the range after scaling
    key_clip = jnp.array([minval * k_scale * 2, maxval * k_scale * 2])
    value_clip = jnp.array([maxval * v_scale * 2, minval * v_scale * 2])
    quantized_key_clip, quantized_value_clip = quantize_kv(
        key_clip, value_clip, kv_cache_quantized_dtype, k_scale, v_scale)

    # Values should be clipped to the min/max of the float8 dtype
    expected_key_clip = jnp.array([minval, maxval], dtype=jnp.float8_e4m3fn)
    expected_value_clip = jnp.array([maxval, minval], dtype=jnp.float8_e4m3fn)

    assert jnp.array_equal(quantized_key_clip, expected_key_clip)
    assert jnp.array_equal(quantized_value_clip, expected_value_clip)
