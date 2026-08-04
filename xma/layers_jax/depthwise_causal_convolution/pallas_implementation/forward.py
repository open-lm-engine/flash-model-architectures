# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from functools import partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from ....math import ceil_divide


def _forward_kernel(x_ref, W_ref, b_ref, h0_ref, y_ref, h_ref, *, BLOCK_SIZE_S: int, S: int, K: int) -> None:
    @pl.when(pl.program_id(1) == 0)
    def _():
        h_ref[...] = h0_ref[...]

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


@partial(jax.jit, static_argnames=("BLOCK_SIZE_S",))
def _depthwise_causal_convolution_forward_core(
    x: jax.Array, W: jax.Array, b: jax.Array, h0: jax.Array, BLOCK_SIZE_S: int
) -> tuple[jax.Array, jax.Array]:
    B, S, H = x.shape
    K = W.shape[0]

    kernel = pl.pallas_call(
        partial(_forward_kernel, BLOCK_SIZE_S=BLOCK_SIZE_S, S=S, K=K),
        out_shape=(
            jax.ShapeDtypeStruct((B, S, H), x.dtype),
            jax.ShapeDtypeStruct((B, h0.shape[1], H), jnp.float32),
        ),
        grid=(B, ceil_divide(S, BLOCK_SIZE_S)),
        in_specs=(
            pl.BlockSpec(block_shape=(None, BLOCK_SIZE_S, H), index_map=lambda b, c: (b, c, 0)),
            pl.BlockSpec(block_shape=(K, H), index_map=lambda b, c: (0, 0)),
            pl.BlockSpec(block_shape=(1, H), index_map=lambda b, c: (0, 0)),
            pl.BlockSpec(block_shape=(None, h0.shape[1], H), index_map=lambda b, c: (b, 0, 0)),
        ),
        out_specs=(
            pl.BlockSpec(block_shape=(None, BLOCK_SIZE_S, H), index_map=lambda b, c: (b, c, 0)),
            pl.BlockSpec(block_shape=(None, h0.shape[1], H), index_map=lambda b, c: (b, 0, 0)),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
    )

    return kernel(x, W, b, h0)


def _depthwise_causal_convolution_forward_pallas(
    x: jax.Array, W: jax.Array, b: jax.Array | None, h0: jax.Array | None, BLOCK_SIZE_S: int
) -> tuple[jax.Array, jax.Array]:
    B, _, H = x.shape
    K = W.shape[-1]

    W = jnp.transpose(W, (1, 0))
    b = jnp.zeros((1, H), dtype=jnp.float32) if b is None else b.astype(jnp.float32)[None, :]

    h0 = (
        jnp.zeros((B, K - 1, H), dtype=x.dtype)
        if h0 is None
        else jnp.transpose(h0[:, :, 1:], (0, 2, 1)).astype(x.dtype)
    )

    y, ht = _depthwise_causal_convolution_forward_core(x=x, W=W, b=b, h0=h0, BLOCK_SIZE_S=BLOCK_SIZE_S)

    return y, ht
