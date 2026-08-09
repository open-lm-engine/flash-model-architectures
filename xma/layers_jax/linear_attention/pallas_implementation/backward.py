# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from functools import partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from ....math import ceil_divide


def _linear_attention_backward_kernel(
    q_ref,
    k_ref,
    v_ref,
    h_checkpoint_ref,
    dy_ref,
    dht_ref,
    dq_ref,
    dk_ref,
    dv_ref,
    dh0_ref,
    dqk_ref,
    dq_term2_ref,
    dk_term2_ref,
    *,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
    S: int,
    V: int,
    NUM_BLOCKS_S: int,
    NUM_V_TILES: int,
) -> None:
    rc = pl.program_id(2)
    vb = pl.program_id(3)

    @pl.when(rc == 0)
    def _():
        if dht_ref is None:
            dh0_ref[...] = jnp.zeros_like(dh0_ref)
        else:
            dh0_ref[...] = dht_ref[...]

    dtype = q_ref.dtype

    BLOCK_ID_S = NUM_BLOCKS_S - 1 - rc
    BLOCK_S = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, 1), 0)
    MASK_S = (BLOCK_ID_S * BLOCK_SIZE_S + BLOCK_S) < S

    q = jnp.where(MASK_S, q_ref[...], 0).astype(dtype)
    k = jnp.where(MASK_S, k_ref[...], 0).astype(dtype)
    v = jnp.where(MASK_S, v_ref[...], 0).astype(dtype)

    dy = jnp.where(MASK_S, dy_ref[...], 0).astype(jnp.float32) * attention_multiplier
    dy = dy.astype(dtype)

    hc = h_checkpoint_ref[...].astype(dtype)
    g = dh0_ref[...]

    row = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, BLOCK_SIZE_S), 0)
    col = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, BLOCK_SIZE_S), 1)
    causal_mask = row >= col

    qk = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
    qk = jnp.where(causal_mask, qk, 0).astype(dtype)

    dv = jax.lax.dot_general(qk, dy, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)
    dv += jax.lax.dot_general(k, g.astype(dtype), (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
    dv_ref[...] = dv.astype(dtype)

    dh0_ref[...] = g + jax.lax.dot_general(q, dy, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)

    BLOCK_V = jax.lax.broadcasted_iota(jnp.int32, (1, v.shape[-1]), 1)
    MASK_V = (vb * v.shape[-1] + BLOCK_V) < V

    v = jnp.where(MASK_V, v, 0)
    dy = jnp.where(MASK_V, dy, 0)
    hc = jnp.where(MASK_V, hc, 0)
    g = jnp.where(MASK_V, g, 0)

    @pl.when(vb == 0)
    def _():
        dqk_ref[...] = jnp.zeros_like(dqk_ref)
        dq_term2_ref[...] = jnp.zeros_like(dq_term2_ref)
        dk_term2_ref[...] = jnp.zeros_like(dk_term2_ref)

    dqk_ref[...] += jax.lax.dot_general(dy, v, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)

    dq_term2_ref[...] += jax.lax.dot_general(dy, hc, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
    dk_term2_ref[...] += jax.lax.dot_general(v, g, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)

    @pl.when(vb == NUM_V_TILES - 1)
    def _():
        dqk = jnp.where(causal_mask, dqk_ref[...], 0).astype(dtype)

        dq = jax.lax.dot_general(dqk, k, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
        dq += dq_term2_ref[...]
        dq_ref[...] = dq.astype(dtype)

        dk = jax.lax.dot_general(dqk, q, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)
        dk += dk_term2_ref[...]
        dk_ref[...] = dk.astype(dtype)


@partial(jax.jit, static_argnames=("attention_multiplier", "BLOCK_SIZE_S", "BLOCK_SIZE_V"))
def _linear_attention_backward_core(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    h: jax.Array,
    dy: jax.Array,
    dht: jax.Array | None,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    B, Nq, S, K = q.shape
    Nk = k.shape[1]
    Nv = v.shape[1]
    V = v.shape[-1]
    N = dy.shape[1]

    Gq = N // Nq
    Gk = N // Nk
    Gv = N // Nv

    NUM_BLOCKS_S = ceil_divide(S, BLOCK_SIZE_S)
    NUM_V_TILES = ceil_divide(V, BLOCK_SIZE_V)

    h_spec = pl.BlockSpec(
        block_shape=(None, None, K, BLOCK_SIZE_V),
        index_map=lambda BLOCK_ID_B, BLOCK_ID_N, BLOCK_ID_S, BLOCK_ID_V: (BLOCK_ID_B, BLOCK_ID_N, 0, BLOCK_ID_V),
    )

    kernel = pl.pallas_call(
        partial(
            _linear_attention_backward_kernel,
            attention_multiplier=attention_multiplier,
            BLOCK_SIZE_S=BLOCK_SIZE_S,
            S=S,
            V=V,
            NUM_BLOCKS_S=NUM_BLOCKS_S,
            NUM_V_TILES=NUM_V_TILES,
        ),
        out_shape=(
            jax.ShapeDtypeStruct(shape=(B, N, S, K), dtype=q.dtype),
            jax.ShapeDtypeStruct(shape=(B, N, S, K), dtype=q.dtype),
            jax.ShapeDtypeStruct(shape=(B, N, S, V), dtype=q.dtype),
            jax.ShapeDtypeStruct(shape=(B, N, K, V), dtype=jnp.float32),
        ),
        grid=(B, N, NUM_BLOCKS_S, NUM_V_TILES),
        in_specs=(
            pl.BlockSpec(
                block_shape=(None, None, BLOCK_SIZE_S, K),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_N, BLOCK_ID_S, BLOCK_ID_V: (
                    BLOCK_ID_B,
                    BLOCK_ID_N // Gq,
                    NUM_BLOCKS_S - 1 - BLOCK_ID_S,
                    0,
                ),
            ),
            pl.BlockSpec(
                block_shape=(None, None, BLOCK_SIZE_S, K),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_N, BLOCK_ID_S, BLOCK_ID_V: (
                    BLOCK_ID_B,
                    BLOCK_ID_N // Gk,
                    NUM_BLOCKS_S - 1 - BLOCK_ID_S,
                    0,
                ),
            ),
            pl.BlockSpec(
                block_shape=(None, None, BLOCK_SIZE_S, BLOCK_SIZE_V),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_N, BLOCK_ID_S, BLOCK_ID_V: (
                    BLOCK_ID_B,
                    BLOCK_ID_N // Gv,
                    NUM_BLOCKS_S - 1 - BLOCK_ID_S,
                    BLOCK_ID_V,
                ),
            ),
            pl.BlockSpec(
                block_shape=(None, None, K, BLOCK_SIZE_V),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_N, BLOCK_ID_S, BLOCK_ID_V: (
                    BLOCK_ID_B,
                    BLOCK_ID_N * NUM_BLOCKS_S + (NUM_BLOCKS_S - 1 - BLOCK_ID_S),
                    0,
                    BLOCK_ID_V,
                ),
            ),
            pl.BlockSpec(
                block_shape=(None, None, BLOCK_SIZE_S, BLOCK_SIZE_V),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_N, BLOCK_ID_S, BLOCK_ID_V: (
                    BLOCK_ID_B,
                    BLOCK_ID_N,
                    NUM_BLOCKS_S - 1 - BLOCK_ID_S,
                    BLOCK_ID_V,
                ),
            ),
            None if dht is None else h_spec,
        ),
        out_specs=(
            pl.BlockSpec(
                block_shape=(None, None, BLOCK_SIZE_S, K),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_N, BLOCK_ID_S, BLOCK_ID_V: (
                    BLOCK_ID_B,
                    BLOCK_ID_N,
                    NUM_BLOCKS_S - 1 - BLOCK_ID_S,
                    0,
                ),
            ),
            pl.BlockSpec(
                block_shape=(None, None, BLOCK_SIZE_S, K),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_N, BLOCK_ID_S, BLOCK_ID_V: (
                    BLOCK_ID_B,
                    BLOCK_ID_N,
                    NUM_BLOCKS_S - 1 - BLOCK_ID_S,
                    0,
                ),
            ),
            pl.BlockSpec(
                block_shape=(None, None, BLOCK_SIZE_S, BLOCK_SIZE_V),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_N, BLOCK_ID_S, BLOCK_ID_V: (
                    BLOCK_ID_B,
                    BLOCK_ID_N,
                    NUM_BLOCKS_S - 1 - BLOCK_ID_S,
                    BLOCK_ID_V,
                ),
            ),
            h_spec,
        ),
        scratch_shapes=[
            pltpu.VMEM((BLOCK_SIZE_S, BLOCK_SIZE_S), jnp.float32),
            pltpu.VMEM((BLOCK_SIZE_S, K), jnp.float32),
            pltpu.VMEM((BLOCK_SIZE_S, K), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary", "arbitrary")),
    )

    return kernel(q, k, v, h, dy, dht)
