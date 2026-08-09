# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from functools import partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from ....math import ceil_divide


def _state_passing_kernel(k_ref, v_ref, h0_ref, h_ref, h_scratch, *, BLOCK_SIZE_S: int, S: int) -> None:
    @pl.when(pl.program_id(3) == 0)
    def _():
        if h0_ref is None:
            h_scratch[...] = jnp.zeros_like(h_scratch)
        else:
            h_scratch[...] = h0_ref[...].astype(jnp.float32)

    dtype = k_ref.dtype

    BLOCK_ID_S = pl.program_id(3)
    BLOCK_S = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, 1), 0)
    MASK_S = (BLOCK_ID_S * BLOCK_SIZE_S + BLOCK_S) < S

    k = jnp.where(MASK_S, k_ref[...], 0).astype(dtype)
    v = jnp.where(MASK_S, v_ref[...], 0).astype(dtype)
    h = h_scratch[...]

    h_ref[...] = h
    h_scratch[...] = h + jax.lax.dot_general(k, v, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)


@partial(jax.jit, static_argnames=("N", "BLOCK_SIZE_S", "BLOCK_SIZE_V"))
def _state_passing_core(
    k: jax.Array, v: jax.Array, h0: jax.Array | None, N: int, BLOCK_SIZE_S: int, BLOCK_SIZE_V: int
) -> jax.Array:
    B, Nk, S, K = k.shape
    Nv = v.shape[1]
    V = v.shape[-1]

    Gk = N // Nk
    Gv = N // Nv

    NUM_BLOCKS_S = ceil_divide(S, BLOCK_SIZE_S)

    kernel = pl.pallas_call(
        partial(_state_passing_kernel, BLOCK_SIZE_S=BLOCK_SIZE_S, S=S),
        out_shape=jax.ShapeDtypeStruct(shape=(B, N * NUM_BLOCKS_S, K, V), dtype=jnp.float32),
        grid=(B, N, ceil_divide(V, BLOCK_SIZE_V), NUM_BLOCKS_S),
        in_specs=(
            pl.BlockSpec(
                block_shape=(None, None, BLOCK_SIZE_S, K),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_N, BLOCK_ID_V, BLOCK_ID_S: (
                    BLOCK_ID_B,
                    BLOCK_ID_N // Gk,
                    BLOCK_ID_S,
                    0,
                ),
            ),
            pl.BlockSpec(
                block_shape=(None, None, BLOCK_SIZE_S, BLOCK_SIZE_V),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_N, BLOCK_ID_V, BLOCK_ID_S: (
                    BLOCK_ID_B,
                    BLOCK_ID_N // Gv,
                    BLOCK_ID_S,
                    BLOCK_ID_V,
                ),
            ),
            (
                None
                if h0 is None
                else pl.BlockSpec(
                    block_shape=(None, None, K, BLOCK_SIZE_V),
                    index_map=lambda BLOCK_ID_B, BLOCK_ID_N, BLOCK_ID_V, BLOCK_ID_S: (
                        BLOCK_ID_B,
                        BLOCK_ID_N,
                        0,
                        BLOCK_ID_V,
                    ),
                )
            ),
        ),
        out_specs=pl.BlockSpec(
            block_shape=(None, None, K, BLOCK_SIZE_V),
            index_map=lambda BLOCK_ID_B, BLOCK_ID_N, BLOCK_ID_V, BLOCK_ID_S: (
                BLOCK_ID_B,
                BLOCK_ID_N * NUM_BLOCKS_S + BLOCK_ID_S,
                0,
                BLOCK_ID_V,
            ),
        ),
        scratch_shapes=[pltpu.VMEM((K, BLOCK_SIZE_V), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "parallel", "arbitrary")),
    )

    return kernel(k, v, h0)
