# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from functools import partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from ....math import ceil_divide


def _state_passing_kernel(x_ref, h0_ref, h_ref, h_scratch, *, K: int) -> None:
    @pl.when(pl.program_id(1) == 0)
    def _():
        h_scratch[...] = h0_ref[...]

    h_ref[...] = h_scratch[...][None]

    BLOCK_SIZE_S = x_ref.shape[0]
    PAD = h_scratch.shape[0]
    offset = PAD - K + 1

    for p in range(K - 1):
        h_scratch[offset + p, :] = x_ref[BLOCK_SIZE_S - K + 1 + p, :]


@partial(jax.jit, static_argnames=("BLOCK_SIZE_S", "K"))
def _state_passing_core(x: jax.Array, h0: jax.Array, BLOCK_SIZE_S: int, K: int) -> jax.Array:
    B, S, H = x.shape
    NUM_BLOCKS_S = ceil_divide(S, BLOCK_SIZE_S)
    PAD = ceil_divide(K - 1, 8) * 8

    h0 = jnp.pad(h0, ((0, 0), (PAD - K + 1, 0), (0, 0)))

    kernel = pl.pallas_call(
        partial(_state_passing_kernel, K=K),
        out_shape=jax.ShapeDtypeStruct((B, NUM_BLOCKS_S, PAD, H), jnp.float32),
        grid=(B, NUM_BLOCKS_S),
        in_specs=(
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, H),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, BLOCK_ID_S, 0),
            ),
            pl.BlockSpec(block_shape=(None, PAD, H), index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, 0, 0)),
        ),
        out_specs=pl.BlockSpec(
            block_shape=(None, 1, PAD, H), index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, BLOCK_ID_S, 0, 0)
        ),
        scratch_shapes=[pltpu.VMEM((PAD, H), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
    )

    return kernel(x, h0)


def _backward_kernel(
    x_ref,
    W_ref,
    ckpt_ref,
    dy_ref,
    dh_last_ref,
    dx_ref,
    dW_ref,
    db_ref,
    dh0_ref,
    dh_scratch,
    *,
    BLOCK_SIZE_S: int,
    S: int,
    K: int,
    PAD: int,
    NUM_BLOCKS_S: int,
) -> None:
    b = pl.program_id(0)
    rc = pl.program_id(1)
    c = NUM_BLOCKS_S - 1 - rc

    H = x_ref.shape[-1]
    dtype = x_ref.dtype
    offset = PAD - K + 1

    @pl.when(rc == 0)
    def _():
        dh_scratch[...] = dh_last_ref[...]

    @pl.when((b == 0) & (rc == 0))
    def _():
        dW_ref[...] = jnp.zeros(dW_ref.shape, dtype=dW_ref.dtype)
        db_ref[...] = jnp.zeros(db_ref.shape, dtype=db_ref.dtype)

    BLOCK_S = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, 1), 0)
    MASK_S = (c * BLOCK_SIZE_S + BLOCK_S) < S

    dy = jnp.where(MASK_S, dy_ref[...], 0).astype(jnp.float32)
    x = jnp.where(MASK_S, x_ref[...], 0).astype(jnp.float32)

    tail_len = BLOCK_SIZE_S - K + 1

    # dx, bulk positions j in [0, tail_len): every tap of the adjoint gather stays inside this chunk's dy.
    dx_tail = jnp.zeros((tail_len, H), dtype=jnp.float32)
    for k in range(K):
        W = W_ref[k, :].astype(jnp.float32)
        dx_tail += W[None, :] * dy[K - 1 - k : K - 1 - k + tail_len, :]

    # dx, boundary positions j in [tail_len, BLOCK_SIZE_S): part of the gradient already arrived from the
    # NEXT chunk (carried in via `dh_scratch`, since this chunk's own trailing K - 1 positions served as
    # that chunk's history), the rest comes from in-range taps of this chunk's own dy.
    dx_boundary_rows = []
    for i in range(K - 1):
        j = tail_len + i
        row = dh_scratch[offset + i, :]
        for k in range(K):
            t = j + (K - 1) - k
            if t < BLOCK_SIZE_S:
                W = W_ref[k, :].astype(jnp.float32)
                row = row + W * dy[t, :]
        dx_boundary_rows.append(row)

    dx = jnp.concatenate([dx_tail, jnp.stack(dx_boundary_rows, axis=0)], axis=0) if K > 1 else dx_tail
    dx_ref[...] = jnp.where(MASK_S, dx, 0).astype(dtype)

    # dW: split each tap's contribution into the part sourced from this chunk's own `x` (a plain, un-
    # concatenated slice pair - safe) and the tiny part sourced from the incoming checkpoint history
    # (scalar rows - also safe).
    for k in range(K):
        x_len = tail_len + k
        dw_k = jnp.sum(dy[K - 1 - k : BLOCK_SIZE_S, :] * x[:x_len, :], axis=0)
        for t in range(K - 1 - k):
            dw_k = dw_k + dy[t, :] * ckpt_ref[0, offset + t + k, :]
        dW_ref[k, :] += dw_k

    db_ref[...] += jnp.sum(dy, axis=0, keepdims=True)

    # gradient into this chunk's own incoming history -> becomes the outgoing `dh_scratch` for the previous
    # (in forward order) chunk, i.e. the next chunk we process since this kernel runs in reverse.
    for p in range(K - 1):
        dh_p = jnp.zeros((H,), dtype=jnp.float32)
        for k in range(p + 1):
            W = W_ref[k, :].astype(jnp.float32)
            dh_p = dh_p + W * dy[p - k, :]
        dh_scratch[offset + p, :] = dh_p

    @pl.when(rc == NUM_BLOCKS_S - 1)
    def _():
        dh0_ref[...] = dh_scratch[...]


@partial(jax.jit, static_argnames=("BLOCK_SIZE_S", "K"))
def _depthwise_causal_convolution_backward_core(
    x: jax.Array,
    W: jax.Array,
    ckpt: jax.Array,
    dy: jax.Array,
    dh_last: jax.Array,
    BLOCK_SIZE_S: int,
    K: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    B, S, H = x.shape
    NUM_BLOCKS_S = ceil_divide(S, BLOCK_SIZE_S)
    PAD = ceil_divide(K - 1, 8) * 8

    dh_last = jnp.pad(dh_last, ((0, 0), (PAD - K + 1, 0), (0, 0)))

    kernel = pl.pallas_call(
        partial(_backward_kernel, BLOCK_SIZE_S=BLOCK_SIZE_S, S=S, K=K, PAD=PAD, NUM_BLOCKS_S=NUM_BLOCKS_S),
        out_shape=(
            jax.ShapeDtypeStruct((B, S, H), x.dtype),
            jax.ShapeDtypeStruct((K, H), jnp.float32),
            jax.ShapeDtypeStruct((1, H), jnp.float32),
            jax.ShapeDtypeStruct((B, PAD, H), jnp.float32),
        ),
        grid=(B, NUM_BLOCKS_S),
        in_specs=(
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, H),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, NUM_BLOCKS_S - 1 - BLOCK_ID_S, 0),
            ),
            pl.BlockSpec(block_shape=(K, H), index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (0, 0)),
            pl.BlockSpec(
                block_shape=(None, 1, PAD, H),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, NUM_BLOCKS_S - 1 - BLOCK_ID_S, 0, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, H),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, NUM_BLOCKS_S - 1 - BLOCK_ID_S, 0),
            ),
            pl.BlockSpec(block_shape=(None, PAD, H), index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, 0, 0)),
        ),
        out_specs=(
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, H),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, NUM_BLOCKS_S - 1 - BLOCK_ID_S, 0),
            ),
            pl.BlockSpec(block_shape=(K, H), index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (0, 0)),
            pl.BlockSpec(block_shape=(1, H), index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (0, 0)),
            pl.BlockSpec(block_shape=(None, PAD, H), index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, 0, 0)),
        ),
        scratch_shapes=[pltpu.VMEM((PAD, H), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
    )

    dx, dW, db, dh0 = kernel(x, W, ckpt, dy, dh_last)

    return dx, dW, db, dh0[:, 1 - K :, :]
