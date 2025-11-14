from typing import Tuple

import jax
import jax.numpy as jnp

MXFP4_BLOCK_SIZE = 32


def quantize_to_mxfp4(tensor: jax.Array,
                      axis: int = -1) -> Tuple[jax.Array, jax.Array]:
    """Quantize a tensor to mxfp4 which has e2m1 weight and e8m0 scale."""

    tensor_q, scale = quantize_tensor(jnp.float4_e2m1fn, tensor, axis,
                                      MXFP4_BLOCK_SIZE)

    # Since TPU does not have native support for e8m0, we convert scale into
    # e8m0 manually and store it as uint8
    e8m0_finfo = jnp.finfo(jnp.float8_e8m0fnu)
    _, scale_exp = jnp.frexp(scale)
    # Subtract exponents by one since e8m0 has no decimal
    scale_exp -= 1
    scale_exp = (scale_exp - e8m0_finfo.minexp).astype(jnp.uint8)

    return tensor_q, scale_exp


def quantize_to_mxfp4_packed(tensor: jax.Array,
                             axis: int = -1) -> Tuple[jax.Array, jax.Array]:
    """Quantize a tensor to mxfp4 and pack it into uint8."""
    tensor_q, scale = quantize_to_mxfp4(tensor, axis)

    # last two e2m1 elements will be packed into a single uint8 element.
    bitcast_shape = tensor_q.shape[:-1] + (-1, 2)
    tensor_q = tensor_q.reshape(bitcast_shape)
    tensor_packed = jax.lax.bitcast_convert_type(tensor_q, jnp.uint8)
    return tensor_packed, scale


def u8_unpack_e2m1(u8_packed_e2m1: jax.Array) -> jax.Array:
    """Unpack e2m1 tensor packed into u8."""
    assert u8_packed_e2m1.dtype == jnp.uint8
    e2m1 = jax.lax.bitcast_convert_type(u8_packed_e2m1, jnp.float4_e2m1fn)
    # bitcast creates one more dimension that splits 8 bits into two e2m1.
    # we flatten them with the last dim.
    return jnp.reshape(e2m1, e2m1.shape[:-2] + (-1, ))


def e8m0_to_fp32(u8: jax.Array) -> jax.Array:
    """Convert e8m0 (that was bitcasted to u8) into fp32"""
    assert u8.dtype == jnp.uint8

    e8_finfo = jnp.finfo(jnp.float8_e8m0fnu)
    exponents = u8.astype(jnp.int32) + e8_finfo.minexp
    ones = jnp.ones_like(u8, dtype=jnp.float32)
    return jnp.ldexp(ones, exponents)


def dequantize_tensor(tensor_q: jax.Array,
                      scale: jax.Array,
                      axis: int = -1,
                      block_size: int | None = None,
                      out_dtype: jnp.dtype = jnp.bfloat16) -> jax.Array:
    if axis == -1:
        axis = len(tensor_q) - 1

    # TODO(kyuyeunk): Programatically figure out block size from the shapes of
    # tensor and scale.
    if block_size is not None:
        orig_shape = tensor_q.shape
        subchannel_shape = orig_shape[:axis] + (
            -1, block_size) + orig_shape[axis + 1:]
        tensor_q = tensor_q.reshape(subchannel_shape)
        scale = jnp.expand_dims(scale, axis)

    tensor = (tensor_q.astype(jnp.float32) * scale).astype(out_dtype)

    if block_size is not None:
        tensor = tensor.reshape(orig_shape)
    return tensor


def dequantize_mxfp4_tensor(tensor_q: jax.Array,
                            scale: jax.Array,
                            axis: int = -1,
                            out_dtype: jnp.dtype = jnp.bfloat16) -> jax.Array:
    tensor_e2m1 = u8_unpack_e2m1(tensor_q)
    scale_fp32 = e8m0_to_fp32(scale)

    return dequantize_tensor(tensor_e2m1, scale_fp32, axis, MXFP4_BLOCK_SIZE,
                             out_dtype)


def quantize_tensor(dtype: jnp.dtype,
                    tensor: jax.Array,
                    axis: int = -1,
                    block_size: int | None = None):
    if axis == -1:
        axis = len(tensor) - 1

    if block_size is not None:
        orig_shape = tensor.shape
        blocked_shape = orig_shape[:axis] + (
            -1, block_size) + orig_shape[axis + 1:]
        tensor = tensor.reshape(blocked_shape)
        axis += 1

    if isinstance(dtype, jnp.floating):
        dtype_info = jnp.finfo(dtype)
    else:
        dtype_info = jnp.iinfo(dtype)

    dtype_max = float(dtype_info.max)
    dtype_min = float(dtype_info.min)

    abs_max = jnp.max(jnp.abs(tensor), axis=axis, keepdims=True)
    scale = abs_max / dtype_max

    tensor_q = jnp.clip(tensor / scale, dtype_min, dtype_max)
    tensor_q = tensor_q.astype(dtype)

    scale = jnp.squeeze(scale, axis).astype(jnp.float32)
    if block_size is not None:
        tensor_q = tensor_q.reshape(orig_shape)

    return tensor_q, scale


def static_quantize_tensor(dtype: jnp.dtype, tensor: jax.Array,
                           scale: float) -> jax.Array:
    if isinstance(dtype, jnp.floating):
        dtype_info = jnp.finfo(dtype)
    else:
        dtype_info = jnp.iinfo(dtype)

    dtype_max = float(dtype_info.max)
    dtype_min = float(dtype_info.min)

    return jnp.clip(tensor / scale, dtype_min, dtype_max).astype(dtype)


def quantize_kv(dtype: jnp.dtype, key: jax.Array, value: jax.Array,
                k_scale: float, v_scale: float) -> Tuple[jax.Array, jax.Array]:
    """Static quantize key and value tensors."""
    key = static_quantize_tensor(dtype, key, k_scale)
    value = static_quantize_tensor(dtype, value, v_scale)
    return key, value
