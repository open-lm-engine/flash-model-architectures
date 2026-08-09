# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from functools import partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from ....math import ceil_divide


def _backward_kernel(
    x_ref,
    W_ref,
    h_ref,
    dy_ref,
    dht_ref,
    dx_ref,
    dW_ref,
    db_ref,
    dh0_ref,
    dy_scratch,
    dht_scratch,
    BLOCK_SIZE_S: int,
    S: int,
    K: int,
    PAD: int,
    NUM_BLOCKS_S: int,
) -> None:
    BLOCK_ID_B = pl.program_id(0)
    BLOCK_ID_S_REVERSE = pl.program_id(1)
    BLOCK_ID_S = NUM_BLOCKS_S - 1 - BLOCK_ID_S_REVERSE

    H = x_ref.shape[-1]
    dtype = x_ref.dtype
    offset = PAD - K + 1

    @pl.when(BLOCK_ID_S_REVERSE == 0)
    def _():
        dy_scratch[...] = jnp.zeros_like(dy_scratch)
        dht_scratch[...] = jnp.zeros_like(dht_scratch)
        if dht_ref is not None:
            dht_scratch[offset:, :] = dht_ref[...]

    @pl.when((BLOCK_ID_B == 0) & (BLOCK_ID_S_REVERSE == 0))
    def _():
        dW_ref[...] = jnp.zeros(dW_ref.shape, dtype=dW_ref.dtype)
        db_ref[...] = jnp.zeros(db_ref.shape, dtype=db_ref.dtype)

    BLOCK_S = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, 1), 0)
    MASK_S = (BLOCK_ID_S * BLOCK_SIZE_S + BLOCK_S) < S

    dy = jnp.where(MASK_S, dy_ref[...], 0).astype(jnp.float32)
    x = jnp.where(MASK_S, x_ref[...], 0).astype(jnp.float32)

    # same windowing trick as the forward, in both directions: dx reads dy forward in time so it needs
    # the next block's leading rows appended, while dW correlates dy against the input so it needs the
    # previous block's trailing rows prepended. Both then become full-block slices at static offsets,
    # replacing the K * (K - 1) single-row reads the boundary rows used to need (a row is 1/8 of a
    # vreg, and dynamic row indexing lowers to vector.extract, which is slow and 32-bit only).
    dy_ext = jnp.concatenate([dy, dy_scratch[...]], axis=0)
    x_ext = jnp.concatenate([h_ref[0].astype(jnp.float32), x], axis=0)

    dx = jnp.zeros((BLOCK_SIZE_S, H), dtype=jnp.float32)
    for k in range(K):
        W = W_ref[k, :].astype(jnp.float32)
        dx += W[None, :] * dy_ext[K - 1 - k : K - 1 - k + BLOCK_SIZE_S, :]
        dW_ref[k, :] += jnp.sum(dy * x_ext[offset + k : offset + k + BLOCK_SIZE_S, :], axis=0)

    db_ref[...] += jnp.sum(dy, axis=0, keepdims=True)
    dx_ref[...] = jnp.where(MASK_S, dx, 0).astype(dtype)

    state_prefix = max(K - 1 - S, 0)
    x_state_start = max(S - (K - 1), 0)

    # ht is a plain slice of the input, so dht lands on dx at fixed positions. Which block owns each
    # position is known at trace time, so every other block now skips these selects entirely instead
    # of running K - 1 of them per block.
    if dht_ref is not None:
        dht_rows = {}
        for p in range(state_prefix, K - 1):
            block_index, row_in_block = divmod(x_state_start + p - state_prefix, BLOCK_SIZE_S)
            dht_rows.setdefault(block_index, []).append((p, row_in_block))

        for block_index, rows in dht_rows.items():

            @pl.when(BLOCK_ID_S == block_index)
            def _(rows=rows):
                update = jnp.zeros((BLOCK_SIZE_S, H), dtype=jnp.float32)
                for p, row_in_block in rows:
                    update += jnp.where(BLOCK_S == row_in_block, dht_scratch[offset + p, :], 0)

                dx_ref[...] += update.astype(dtype)

    @pl.when(BLOCK_ID_S == 0)
    def _():
        # dh0[offset + p] = sum_{k <= p} W[k] * dy[p - k]; zero-padding dy on the left drops the
        # out-of-range taps on its own, so this is the same fixed-offset slice pattern as above
        dy_padded = jnp.concatenate([jnp.zeros((PAD, H), dtype=jnp.float32), dy_ext], axis=0)

        dh0 = jnp.zeros((PAD, H), dtype=jnp.float32)
        for k in range(K):
            W = W_ref[k, :].astype(jnp.float32)
            dh0 += W[None, :] * dy_padded[K - 1 - k : K - 1 - k + PAD, :]

        dh0_ref[...] = dh0

        # when S < K - 1, x is too short to fill ht, so ht keeps the tail of h0: ht[p] == h0[S + p]
        # for p < state_prefix, and dht flows straight back to that row
        for p in range(state_prefix):
            dh0_ref[offset + S + p, :] += dht_scratch[offset + p, :]

    # hand this block's leading rows to the block before it (the grid walks S in reverse)
    dy_scratch[...] = dy[:PAD, :]


@partial(jax.jit, static_argnames=("BLOCK_SIZE_S", "K"))
def _backward_core(
    x: jax.Array,
    W: jax.Array,
    h: jax.Array,
    dy: jax.Array,
    dht: jax.Array | None,
    BLOCK_SIZE_S: int,
    K: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    B, S, H = x.shape
    NUM_BLOCKS_S = ceil_divide(S, BLOCK_SIZE_S)
    PAD = ceil_divide(K - 1, 8) * 8
    state_size = K - 1

    # a block has to be long enough to carry the whole history in its leading/trailing PAD rows
    assert BLOCK_SIZE_S >= PAD, f"BLOCK_SIZE_S ({BLOCK_SIZE_S}) must be >= {PAD} for kernel_size ({K})"

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
            (
                None
                if dht is None
                else pl.BlockSpec(
                    block_shape=(None, state_size, H),
                    index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, 0, 0),
                )
            ),
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
        scratch_shapes=[pltpu.VMEM((PAD, H), jnp.float32), pltpu.VMEM((PAD, H), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
    )

    dx, dW, db, dh0 = kernel(x, W, h, dy, dht)

    return dx, dW, db, dh0
