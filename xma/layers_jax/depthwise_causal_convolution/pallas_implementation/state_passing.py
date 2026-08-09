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
        if h0_ref is None:
            h_scratch[...] = jnp.zeros_like(h_scratch)
        else:
            h_scratch[...] = h0_ref[...].astype(jnp.float32)

    h_ref[...] = h_scratch[...][None]

    BLOCK_SIZE_S = x_ref.shape[0]
    PAD = h_scratch.shape[0]
    offset = PAD - K + 1

    # Mosaic's vector.extract (used for single-row indexing below) only supports 32-bit
    # element types, so extract rows from a float32 copy rather than the bf16 `x_ref`.
    x_f32 = x_ref[...].astype(jnp.float32)
    for p in range(K - 1):
        h_scratch[offset + p, :] = x_f32[BLOCK_SIZE_S - K + 1 + p, :]


@partial(jax.jit, static_argnames=("BLOCK_SIZE_S", "K"))
def _state_passing_core(x: jax.Array, h0: jax.Array | None, BLOCK_SIZE_S: int, K: int) -> jax.Array:
    B, S, H = x.shape
    NUM_BLOCKS_S = ceil_divide(S, BLOCK_SIZE_S)
    PAD = ceil_divide(K - 1, 8) * 8

    kernel = pl.pallas_call(
        partial(_state_passing_kernel, K=K),
        out_shape=jax.ShapeDtypeStruct((B, NUM_BLOCKS_S, PAD, H), jnp.float32),
        grid=(B, NUM_BLOCKS_S),
        in_specs=(
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, H),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, BLOCK_ID_S, 0),
            ),
            (
                None
                if h0 is None
                else pl.BlockSpec(
                    block_shape=(None, PAD, H), index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, 0, 0)
                )
            ),
        ),
        out_specs=pl.BlockSpec(
            block_shape=(None, 1, PAD, H), index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, BLOCK_ID_S, 0, 0)
        ),
        scratch_shapes=[pltpu.VMEM((PAD, H), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
    )

    return kernel(x, h0)
