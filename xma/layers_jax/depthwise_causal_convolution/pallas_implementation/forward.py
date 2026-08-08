# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from functools import partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from ....math import ceil_divide


def _forward_kernel(
    x_ref,
    W_ref,
    b_ref,
    h0_ref,
    y_ref,
    h_scratch,
    BLOCK_SIZE_S: int,
    S: int,
    K: int,
    PAD: int,
) -> None:
    @pl.when(pl.program_id(1) == 0)
    def _():
        if h0_ref is None:
            h_scratch[...] = jnp.zeros_like(h_scratch)
        else:
            h_scratch[...] = h0_ref[...].astype(jnp.float32)

    dtype = x_ref.dtype
    H = x_ref.shape[-1]

    BLOCK_ID_S = pl.program_id(1)
    BLOCK_S = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, 1), 0)
    MASK_S = (BLOCK_ID_S * BLOCK_SIZE_S + BLOCK_S) < S

    x = jnp.where(MASK_S, x_ref[...], 0).astype(dtype)
    b = jnp.zeros((1, H), dtype=jnp.float32) if b_ref is None else b_ref[...].astype(jnp.float32)

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
            W = W_ref[k, :].astype(jnp.float32)

            p = j + k
            if p < K - 1:
                source = h_scratch[offset + p, :]
            else:
                source = x[p - K + 1, :].astype(jnp.float32)

            row += W * source
        head_rows.append(row)

    y = jnp.concatenate([jnp.stack(head_rows, axis=0), y_tail], axis=0)
    y_ref[...] = y.astype(dtype)

    for p in range(K - 1):
        h_scratch[offset + p, :] = x[tail_len + p, :].astype(jnp.float32)


@partial(jax.jit, static_argnames=("BLOCK_SIZE_S",))
def _forward_core(
    x: jax.Array,
    W: jax.Array,
    b: jax.Array | None,
    h0: jax.Array | None,
    BLOCK_SIZE_S: int,
) -> jax.Array:
    B, S, H = x.shape
    K = W.shape[0]
    PAD = ceil_divide(K - 1, 8) * 8

    x_spec = pl.BlockSpec(
        block_shape=(None, BLOCK_SIZE_S, H), index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, BLOCK_ID_S, 0)
    )

    kernel = pl.pallas_call(
        partial(_forward_kernel, BLOCK_SIZE_S=BLOCK_SIZE_S, S=S, K=K, PAD=PAD),
        out_shape=jax.ShapeDtypeStruct((B, S, H), x.dtype),
        grid=(B, ceil_divide(S, BLOCK_SIZE_S)),
        in_specs=(
            x_spec,
            pl.BlockSpec(block_shape=(K, H), index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (0, 0)),
            None if b is None else pl.BlockSpec(block_shape=(1, H), index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (0, 0)),
            (
                None
                if h0 is None
                else pl.BlockSpec(
                    block_shape=(None, PAD, H), index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, 0, 0)
                )
            ),
        ),
        out_specs=x_spec,
        scratch_shapes=[pltpu.VMEM((PAD, H), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
    )

    return kernel(x, W, b, h0)
