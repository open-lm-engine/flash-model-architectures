# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from functools import partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from ....math import ceil_divide


def _forward(x_ref, W_ref, b_ref, h_ref, y_ref, *, BLOCK_SIZE_S: int, S: int, K: int) -> None:
    dtype = x_ref.dtype
    H = x_ref.shape[-1]

    BLOCK_ID_S = pl.program_id(1)
    BLOCK_S = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, 1), 0)
    MASK_S = (BLOCK_ID_S * BLOCK_SIZE_S + BLOCK_S) < S

    x = jnp.where(MASK_S, x_ref[...], 0).astype(dtype)
    x = jnp.concatenate([h_ref[...], x], axis=0)

    b = b_ref[...].astype(jnp.float32)
    y = jnp.zeros((BLOCK_SIZE_S, H), dtype=jnp.float32) + b

    for k in range(K):
        W = W_ref[k, :].astype(jnp.float32)
        y += W[None, :] * x[k : k + BLOCK_SIZE_S, :].astype(jnp.float32)

    y_ref[...] = y.astype(dtype)
    h_ref[...] = x[1 - K :, :]


def _forward_kernel(x_ref, W_ref, b_ref, h0_ref, y_ref, h_ref, *, BLOCK_SIZE_S: int, S: int, K: int) -> None:
    @pl.when(pl.program_id(1) == 0)
    def _():
        h_ref[...] = h0_ref[...]

    _forward(x_ref=x_ref, W_ref=W_ref, b_ref=b_ref, h_ref=h_ref, y_ref=y_ref, BLOCK_SIZE_S=BLOCK_SIZE_S, S=S, K=K)


def _forward_zero_h0_kernel(x_ref, W_ref, b_ref, y_ref, h_ref, *, BLOCK_SIZE_S: int, S: int, K: int) -> None:
    @pl.when(pl.program_id(1) == 0)
    def _():
        h_ref[...] = jnp.zeros_like(h_ref)

    _forward(x_ref=x_ref, W_ref=W_ref, b_ref=b_ref, h_ref=h_ref, y_ref=y_ref, BLOCK_SIZE_S=BLOCK_SIZE_S, S=S, K=K)


@partial(jax.jit, static_argnames=("BLOCK_SIZE_S",))
def _depthwise_causal_convolution_forward_core(
    x: jax.Array, W: jax.Array, b: jax.Array, h0: jax.Array | None, BLOCK_SIZE_S: int
) -> tuple[jax.Array, jax.Array]:
    B, S, H = x.shape
    K = W.shape[0]

    x_spec = pl.BlockSpec(block_shape=(None, BLOCK_SIZE_S, H), index_map=lambda b, c: (b, c, 0))
    W_spec = pl.BlockSpec(block_shape=(K, H), index_map=lambda b, c: (0, 0))
    b_spec = pl.BlockSpec(block_shape=(1, H), index_map=lambda b, c: (0, 0))
    h_spec = pl.BlockSpec(block_shape=(None, K - 1, H), index_map=lambda b, c: (b, 0, 0))

    if h0 is None:
        kernel_fn = _forward_zero_h0_kernel
        in_specs = (x_spec, W_spec, b_spec)
        args = (x, W, b)
    else:
        kernel_fn = _forward_kernel
        in_specs = (x_spec, W_spec, b_spec, h_spec)
        args = (x, W, b, h0)

    kernel = pl.pallas_call(
        partial(kernel_fn, BLOCK_SIZE_S=BLOCK_SIZE_S, S=S, K=K),
        out_shape=(
            jax.ShapeDtypeStruct((B, S, H), x.dtype),
            jax.ShapeDtypeStruct((B, K - 1, H), jnp.float32),
        ),
        grid=(B, ceil_divide(S, BLOCK_SIZE_S)),
        in_specs=in_specs,
        out_specs=(x_spec, h_spec),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
    )

    return kernel(*args)


def _depthwise_causal_convolution_forward_pallas(
    x: jax.Array, W: jax.Array, b: jax.Array | None, h0: jax.Array | None, BLOCK_SIZE_S: int
) -> tuple[jax.Array, jax.Array]:
    H = x.shape[-1]

    W = jnp.transpose(W, (1, 0))
    b = jnp.zeros((1, H), dtype=jnp.float32) if b is None else b.astype(jnp.float32)[None, :]

    if h0 is not None:
        h0 = jnp.transpose(h0[:, :, 1:], (0, 2, 1)).astype(x.dtype)

    y, ht = _depthwise_causal_convolution_forward_core(x=x, W=W, b=b, h0=h0, BLOCK_SIZE_S=BLOCK_SIZE_S)

    return y, ht
