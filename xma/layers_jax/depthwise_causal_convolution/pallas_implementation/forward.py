# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from functools import partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from ....math import ceil_divide


def _forward(x_ref, W_ref, b_ref, h_ref, y_ref, *, BLOCK_SIZE_S: int, S: int, K: int, PAD: int) -> None:
    dtype = x_ref.dtype
    H = x_ref.shape[-1]

    BLOCK_ID_S = pl.program_id(1)
    BLOCK_S = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, 1), 0)
    MASK_S = (BLOCK_ID_S * BLOCK_SIZE_S + BLOCK_S) < S

    x = jnp.where(MASK_S, x_ref[...], 0).astype(dtype)
    b = b_ref[...].astype(jnp.float32)

    tail_len = BLOCK_SIZE_S - K + 1
    y_tail = jnp.zeros((tail_len, H), dtype=jnp.float32) + b
    for k in range(K):
        W = W_ref[k, :].astype(jnp.float32)
        y_tail += W[None, :] * x[k : k + tail_len, :].astype(jnp.float32)

    offset = PAD - K + 1

    head_rows = []
    for j in range(K - 1):
        row = b[0]
        for k in range(K):
            p = j + k
            W = W_ref[k, :].astype(jnp.float32)
            source = h_ref[offset + p, :] if p < K - 1 else x[p - K + 1, :].astype(jnp.float32)
            row = row + W * source
        head_rows.append(row)

    y = jnp.concatenate([jnp.stack(head_rows, axis=0), y_tail], axis=0)
    y_ref[...] = y.astype(dtype)

    for p in range(K - 1):
        h_ref[offset + p, :] = x[tail_len + p, :]


def _forward_kernel(x_ref, W_ref, b_ref, h0_ref, y_ref, h_ref, *, BLOCK_SIZE_S: int, S: int, K: int, PAD: int) -> None:
    @pl.when(pl.program_id(1) == 0)
    def _():
        h_ref[...] = h0_ref[...]

    _forward(
        x_ref=x_ref, W_ref=W_ref, b_ref=b_ref, h_ref=h_ref, y_ref=y_ref, BLOCK_SIZE_S=BLOCK_SIZE_S, S=S, K=K, PAD=PAD
    )


def _forward_zero_h0_kernel(x_ref, W_ref, b_ref, y_ref, h_ref, *, BLOCK_SIZE_S: int, S: int, K: int, PAD: int) -> None:
    @pl.when(pl.program_id(1) == 0)
    def _():
        h_ref[...] = jnp.zeros_like(h_ref)

    _forward(
        x_ref=x_ref, W_ref=W_ref, b_ref=b_ref, h_ref=h_ref, y_ref=y_ref, BLOCK_SIZE_S=BLOCK_SIZE_S, S=S, K=K, PAD=PAD
    )


@partial(jax.jit, static_argnames=("BLOCK_SIZE_S",))
def _forward_core(
    x: jax.Array, W: jax.Array, b: jax.Array, h0: jax.Array | None, BLOCK_SIZE_S: int
) -> tuple[jax.Array, jax.Array]:
    B, S, H = x.shape
    K = W.shape[0]
    PAD = ceil_divide(K - 1, 8) * 8

    x_spec = pl.BlockSpec(
        block_shape=(None, BLOCK_SIZE_S, H), index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, BLOCK_ID_S, 0)
    )

    W_spec = pl.BlockSpec(block_shape=(K, H), index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (0, 0))
    b_spec = pl.BlockSpec(block_shape=(1, H), index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (0, 0))
    h_spec = pl.BlockSpec(block_shape=(None, PAD, H), index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, 0, 0))

    if h0 is None:
        kernel_fn = _forward_zero_h0_kernel
        in_specs = (x_spec, W_spec, b_spec)
        args = (x, W, b)
    else:
        kernel_fn = _forward_kernel
        h0 = jnp.pad(h0, ((0, 0), (PAD - K + 1, 0), (0, 0)))
        in_specs = (x_spec, W_spec, b_spec, h_spec)
        args = (x, W, b, h0)

    kernel = pl.pallas_call(
        partial(kernel_fn, BLOCK_SIZE_S=BLOCK_SIZE_S, S=S, K=K, PAD=PAD),
        out_shape=(
            jax.ShapeDtypeStruct((B, S, H), x.dtype),
            jax.ShapeDtypeStruct((B, PAD, H), jnp.float32),
        ),
        grid=(B, ceil_divide(S, BLOCK_SIZE_S)),
        in_specs=in_specs,
        out_specs=(x_spec, h_spec),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
    )

    y, ht = kernel(*args)

    return y, ht[:, 1 - K :, :]
