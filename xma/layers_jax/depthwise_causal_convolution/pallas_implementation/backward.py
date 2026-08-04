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


def _checkpoint_kernel(x_ref, h0_ref, ckpt_ref, h_ref, *, K: int) -> None:
    @pl.when(pl.program_id(1) == 0)
    def _():
        h_ref[...] = h0_ref[...]

    ckpt_ref[...] = h_ref[...][None]

    x = jnp.concatenate([h_ref[...], x_ref[...]], axis=0)
    h_ref[...] = x[1 - K :, :]


@partial(jax.jit, static_argnames=("BLOCK_SIZE_S", "K"))
def _depthwise_causal_convolution_checkpoint_core(x: jax.Array, h0: jax.Array, BLOCK_SIZE_S: int, K: int) -> jax.Array:
    # x: (B, S, H); h0: (B, K - 1, H) -> checkpoints: (B, NUM_BLOCKS_S, K - 1, H)
    B, S, H = x.shape
    NUM_BLOCKS_S = ceil_divide(S, BLOCK_SIZE_S)

    kernel = pl.pallas_call(
        partial(_checkpoint_kernel, K=K),
        out_shape=jax.ShapeDtypeStruct((B, NUM_BLOCKS_S, K - 1, H), jnp.float32),
        grid=(B, NUM_BLOCKS_S),
        in_specs=(
            pl.BlockSpec(block_shape=(None, BLOCK_SIZE_S, H), index_map=lambda b, c: (b, c, 0)),
            pl.BlockSpec(block_shape=(None, K - 1, H), index_map=lambda b, c: (b, 0, 0)),
        ),
        out_specs=pl.BlockSpec(block_shape=(None, 1, K - 1, H), index_map=lambda b, c: (b, c, 0, 0)),
        scratch_shapes=[pltpu.VMEM((K - 1, H), jnp.float32)],
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
    NUM_BLOCKS_S: int,
) -> None:
    # grid is (B, rc) with rc a reverse counter (rc == 0 is the LAST chunk in time). dW/db are global
    # reductions over every (batch, position) - they use a constant index_map across the whole grid and
    # accumulate via the same revisit pattern used elsewhere in this codebase, reset once at the very first
    # grid step (b == 0 and rc == 0), not once per batch. `dh_scratch` carries, in reverse order, the
    # gradient w.r.t. the (K - 1, H) history a chunk received from the chunk before it in time; it resets
    # once per batch (rc == 0) since it is a per-batch running accumulator, and its final value (after the
    # first, i.e. earliest, chunk) is exactly dh0 - the gradient into `h0`.
    b = pl.program_id(0)
    rc = pl.program_id(1)
    c = NUM_BLOCKS_S - 1 - rc

    H = x_ref.shape[-1]
    dtype = x_ref.dtype

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
    x = jnp.where(MASK_S, x_ref[...], 0).astype(dtype)

    combined_width = K - 1 + BLOCK_SIZE_S
    x = jnp.concatenate([ckpt_ref[...][0], x], axis=0)
    dx = jnp.zeros((combined_width, H), dtype=jnp.float32)

    for k in range(K):
        W = W_ref[k, :].astype(jnp.float32)
        dW_ref[k, :] += jnp.sum(dy * x[k : k + BLOCK_SIZE_S, :].astype(jnp.float32), axis=0)
        # adjoint of forward's shifted-slice gather: dx[k : k + BLOCK] += W[k] * dy. NOTE: `.at[].add()` with
        # a static slice spanning the WHOLE array triggers a JAX scatter-lowering bug inside Pallas kernels
        # ("captures constants [ShapedArray(int32[0])]") - zero-pad-then-sum is pure concatenation (no
        # scatter/gather), so it sidesteps that entirely.
        dx += jnp.pad(W[None, :] * dy, ((k, combined_width - BLOCK_SIZE_S - k), (0, 0)))

    db_ref[...] += jnp.sum(dy, axis=0, keepdims=True)

    # incoming `dh_scratch` is the gradient w.r.t. THIS chunk's own last (K - 1) positions (they served as
    # the history input for the chunk that follows in time, already processed since we run in reverse) - it
    # lands in `dx`'s TAIL (which becomes dx_ref), not its head. `dx`'s head is this chunk's gradient w.r.t.
    # its OWN incoming history and becomes the new outgoing dh_scratch, untouched by the incoming one since
    # they refer to disjoint position ranges.
    dx = dx.at[-(K - 1) :, :].add(dh_scratch[...])
    dh_scratch[...] = dx[: K - 1, :]
    dx_ref[...] = jnp.where(MASK_S, dx[K - 1 :, :], 0).astype(dtype)

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
    # x, dy: (B, S, H); W: (K, H); ckpt: (B, NUM_BLOCKS_S, K - 1, H); dh_last: (B, K - 1, H), the incoming
    # gradient on the final running state (0 if the caller doesn't chain a further call off of it).
    B, S, H = x.shape
    NUM_BLOCKS_S = ceil_divide(S, BLOCK_SIZE_S)

    kernel = pl.pallas_call(
        partial(_backward_kernel, BLOCK_SIZE_S=BLOCK_SIZE_S, S=S, K=K, NUM_BLOCKS_S=NUM_BLOCKS_S),
        out_shape=(
            jax.ShapeDtypeStruct((B, S, H), x.dtype),
            jax.ShapeDtypeStruct((K, H), jnp.float32),
            jax.ShapeDtypeStruct((1, H), jnp.float32),
            jax.ShapeDtypeStruct((B, K - 1, H), jnp.float32),
        ),
        grid=(B, NUM_BLOCKS_S),
        in_specs=(
            pl.BlockSpec(block_shape=(None, BLOCK_SIZE_S, H), index_map=lambda b, rc: (b, NUM_BLOCKS_S - 1 - rc, 0)),
            pl.BlockSpec(block_shape=(K, H), index_map=lambda b, rc: (0, 0)),
            pl.BlockSpec(block_shape=(None, 1, K - 1, H), index_map=lambda b, rc: (b, NUM_BLOCKS_S - 1 - rc, 0, 0)),
            pl.BlockSpec(block_shape=(None, BLOCK_SIZE_S, H), index_map=lambda b, rc: (b, NUM_BLOCKS_S - 1 - rc, 0)),
            pl.BlockSpec(block_shape=(None, K - 1, H), index_map=lambda b, rc: (b, 0, 0)),
        ),
        out_specs=(
            pl.BlockSpec(block_shape=(None, BLOCK_SIZE_S, H), index_map=lambda b, rc: (b, NUM_BLOCKS_S - 1 - rc, 0)),
            pl.BlockSpec(block_shape=(K, H), index_map=lambda b, rc: (0, 0)),
            pl.BlockSpec(block_shape=(1, H), index_map=lambda b, rc: (0, 0)),
            pl.BlockSpec(block_shape=(None, K - 1, H), index_map=lambda b, rc: (b, 0, 0)),
        ),
        scratch_shapes=[pltpu.VMEM((K - 1, H), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
    )

    return kernel(x, W, ckpt, dy, dh_last)


def _depthwise_causal_convolution_backward_pallas(
    x: jax.Array,
    W: jax.Array,
    h0: jax.Array | None,
    dy: jax.Array,
    dh_last: jax.Array | None,
    BLOCK_SIZE_S: int,
) -> tuple[jax.Array, jax.Array, jax.Array | None, jax.Array | None]:
    B, _, H = x.shape
    K = W.shape[-1]

    W = jnp.transpose(W, (1, 0))

    if h0 is None:
        h0_in = jnp.zeros((B, K - 1, H), dtype=x.dtype)
    else:
        h0_in = jnp.transpose(h0[:, :, 1:], (0, 2, 1)).astype(x.dtype)

    ckpt = _depthwise_causal_convolution_checkpoint_core(x=x, h0=h0_in, BLOCK_SIZE_S=BLOCK_SIZE_S, K=K)

    dh_last_in = jnp.zeros((B, K - 1, H), dtype=jnp.float32) if dh_last is None else dh_last.astype(jnp.float32)

    dx, dW, db, dh0 = _depthwise_causal_convolution_backward_core(
        x, W, ckpt, dy, dh_last_in, BLOCK_SIZE_S=BLOCK_SIZE_S, K=K
    )

    dW = jnp.transpose(dW, (1, 0))
    db = db[0]

    if h0 is None:
        dh0 = None
    else:
        dh0 = jnp.pad(jnp.transpose(dh0, (0, 2, 1)), ((0, 0), (0, 0), (1, 0)))

    return dx, dW, db, dh0
