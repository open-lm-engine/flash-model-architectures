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
from .forward import _state_update


def _checkpoint_kernel(k_ref, v_ref, h0_ref, h_checkpoint_ref, h_ref, *, BLOCK_SIZE_S: int, S: int) -> None:
    # same reasoning as the forward kernel: V is a plain parallel tile here, so the body is unchanged from
    # the untiled version - only grid/block shapes differ. S stays the fastest-varying (last) grid axis.
    @pl.when(pl.program_id(3) == 0)
    def _():
        h_ref[...] = h0_ref[...]

    dtype = k_ref.dtype

    BLOCK_ID_S = pl.program_id(3)
    BLOCK_S = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, 1), 0)
    MASK_S = (BLOCK_ID_S * BLOCK_SIZE_S + BLOCK_S) < S

    k = jnp.where(MASK_S, k_ref[...], 0).astype(dtype)
    v = jnp.where(MASK_S, v_ref[...], 0).astype(dtype)
    h = h_ref[...]

    h_checkpoint_ref[...] = h
    h_ref[...] = _state_update(h=h, k=k, v=v)


def _backward_kernel(
    q_ref,
    k_ref,
    v_ref,
    dy_ref,
    h_checkpoint_ref,
    dh_ref,
    dq_ref,
    dk_ref,
    dv_ref,
    dh0_ref,
    d_masked_qk_scratch,
    dq_term2_scratch,
    dk_term2_scratch,
    *,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
    S: int,
    V: int,
    NUM_BLOCKS_S: int,
    NUM_V_TILES: int,
) -> None:
    # grid is (B, N, S_chunks[reverse], V_tiles). `dv` and the running state-gradient `dh0`/`g` are
    # column-wise independent in V (same as forward's h/y), so they're written directly, once per V-tile.
    # `dq`/`dk` are not: each has a term that reduces over the *full* V dimension (`dy @ v^T` for the
    # shared d_qk term, plus `dy @ h_c^T` for dq and `v @ g^T` for dk), so those terms are accumulated
    # across the V-tile sweep into scratch and only finalized into dq_ref/dk_ref on the last V-tile of
    # each S-chunk. V must be the fastest-varying (last) grid axis for that per-chunk accumulation to see
    # consecutive V-tiles; S (rc) accumulates dh0/g non-adjacently across chunks instead (revisited once
    # per V-tile-sweep), which is exactly what "arbitrary" grid semantics guarantee is still correct.
    rc = pl.program_id(2)
    vb = pl.program_id(3)

    @pl.when(rc == 0)
    def _():
        dh0_ref[...] = dh_ref[...]

    dtype = q_ref.dtype

    physical_chunk = NUM_BLOCKS_S - 1 - rc
    BLOCK_S = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, 1), 0)
    MASK_S = (physical_chunk * BLOCK_SIZE_S + BLOCK_S) < S

    q = jnp.where(MASK_S, q_ref[...], 0).astype(dtype)
    k = jnp.where(MASK_S, k_ref[...], 0).astype(dtype)
    v = jnp.where(MASK_S, v_ref[...], 0).astype(dtype)
    dy = jnp.where(MASK_S, dy_ref[...], 0).astype(jnp.float32) * attention_multiplier
    dy = dy.astype(dtype)

    h_c = h_checkpoint_ref[...].astype(dtype)
    g = dh0_ref[...]

    causal_row_ids = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, BLOCK_SIZE_S), 0)
    causal_col_ids = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, BLOCK_SIZE_S), 1)
    causal_mask = causal_row_ids >= causal_col_ids

    qk = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
    masked_qk = jnp.where(causal_mask, qk, 0).astype(dtype)

    dv = jax.lax.dot_general(masked_qk, dy, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)
    dv += jax.lax.dot_general(k, g.astype(dtype), (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
    dv_ref[...] = dv.astype(dtype)

    dh0_ref[...] = g + jax.lax.dot_general(q, dy, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)

    # V may not divide BLOCK_SIZE_V evenly; out-of-bounds columns of the last V-tile can hold garbage.
    # That's harmless above (garbage only ever lands in a column that's discarded on write-back), but
    # every term below *sums over* the V axis, so a single NaN would poison the whole reduction
    # (0 * NaN = NaN) unless explicitly zeroed first.
    BLOCK_V = jax.lax.broadcasted_iota(jnp.int32, (1, v.shape[-1]), 1)
    MASK_V = (vb * v.shape[-1] + BLOCK_V) < V
    v = jnp.where(MASK_V, v, 0)
    dy = jnp.where(MASK_V, dy, 0)
    h_c = jnp.where(MASK_V, h_c, 0)
    g = jnp.where(MASK_V, g, 0)

    @pl.when(vb == 0)
    def _():
        d_masked_qk_scratch[...] = jnp.zeros_like(d_masked_qk_scratch)
        dq_term2_scratch[...] = jnp.zeros_like(dq_term2_scratch)
        dk_term2_scratch[...] = jnp.zeros_like(dk_term2_scratch)

    d_masked_qk_scratch[...] += jax.lax.dot_general(
        dy, v, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
    )
    dq_term2_scratch[...] += jax.lax.dot_general(dy, h_c, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
    dk_term2_scratch[...] += jax.lax.dot_general(v, g, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)

    @pl.when(vb == NUM_V_TILES - 1)
    def _():
        d_qk = jnp.where(causal_mask, d_masked_qk_scratch[...], 0).astype(dtype)

        dq = jax.lax.dot_general(d_qk, k, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
        dq += dq_term2_scratch[...]
        dq_ref[...] = dq.astype(dtype)

        dk = jax.lax.dot_general(d_qk, q, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)
        dk += dk_term2_scratch[...]
        dk_ref[...] = dk.astype(dtype)


@partial(jax.jit, static_argnames=("BLOCK_SIZE_S", "BLOCK_SIZE_V"))
def _linear_attention_checkpoint_core(
    k: jax.Array, v: jax.Array, h0: jax.Array, BLOCK_SIZE_S: int, BLOCK_SIZE_V: int
) -> tuple[jax.Array, jax.Array]:
    # k, v are already transposed to (B, Nk/Nv, S, K/V); h0 is (B, N, K, V), already defaulted (never None).
    B, Nk, S, K = k.shape
    Nv, V = v.shape[1], v.shape[-1]
    N = h0.shape[1]

    Gk = N // Nk
    Gv = N // Nv

    NUM_BLOCKS_S = ceil_divide(S, BLOCK_SIZE_S)

    k_in_spec = pl.BlockSpec(
        block_shape=(None, None, BLOCK_SIZE_S, K), index_map=lambda b, n, vb, c: (b, n // Gk, c, 0)
    )
    v_in_spec = pl.BlockSpec(
        block_shape=(None, None, BLOCK_SIZE_S, BLOCK_SIZE_V), index_map=lambda b, n, vb, c: (b, n // Gv, c, vb)
    )
    h_running_spec = pl.BlockSpec(
        block_shape=(None, None, K, BLOCK_SIZE_V), index_map=lambda b, n, vb, c: (b, n, 0, vb)
    )

    kernel = pl.pallas_call(
        partial(_checkpoint_kernel, BLOCK_SIZE_S=BLOCK_SIZE_S, S=S),
        out_shape=(
            jax.ShapeDtypeStruct(shape=(B, N * NUM_BLOCKS_S, K, V), dtype=jnp.float32),
            jax.ShapeDtypeStruct(shape=(B, N, K, V), dtype=jnp.float32),
        ),
        grid=(B, N, ceil_divide(V, BLOCK_SIZE_V), NUM_BLOCKS_S),
        in_specs=[k_in_spec, v_in_spec, h_running_spec],
        out_specs=(
            pl.BlockSpec(
                block_shape=(None, None, K, BLOCK_SIZE_V),
                index_map=lambda b, n, vb, c: (b, n * NUM_BLOCKS_S + c, 0, vb),
            ),
            h_running_spec,
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "parallel", "arbitrary")),
    )

    return kernel(k, v, h0)


@partial(jax.jit, static_argnames=("attention_multiplier", "BLOCK_SIZE_S", "BLOCK_SIZE_V"))
def _linear_attention_backward_core(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    dy: jax.Array,
    h_checkpoints: jax.Array,
    dh: jax.Array,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    # q, k, v, dy are already transposed to (B, Nq/Nk/Nv/N, S, K/V); dh is (B, N, K, V), already defaulted.
    B, Nq, S, K = q.shape
    Nk = k.shape[1]
    Nv, V = v.shape[1], v.shape[-1]
    N = dy.shape[1]

    Gq = N // Nq
    Gk = N // Nk
    Gv = N // Nv

    NUM_BLOCKS_S = ceil_divide(S, BLOCK_SIZE_S)
    NUM_V_TILES = ceil_divide(V, BLOCK_SIZE_V)

    h_running_spec = pl.BlockSpec(
        block_shape=(None, None, K, BLOCK_SIZE_V), index_map=lambda b, n, rc, vb: (b, n, 0, vb)
    )

    kernel = pl.pallas_call(
        partial(
            _backward_kernel,
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
        in_specs=[
            pl.BlockSpec(
                block_shape=(None, None, BLOCK_SIZE_S, K),
                index_map=lambda b, n, rc, vb: (b, n // Gq, NUM_BLOCKS_S - 1 - rc, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, None, BLOCK_SIZE_S, K),
                index_map=lambda b, n, rc, vb: (b, n // Gk, NUM_BLOCKS_S - 1 - rc, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, None, BLOCK_SIZE_S, BLOCK_SIZE_V),
                index_map=lambda b, n, rc, vb: (b, n // Gv, NUM_BLOCKS_S - 1 - rc, vb),
            ),
            pl.BlockSpec(
                block_shape=(None, None, BLOCK_SIZE_S, BLOCK_SIZE_V),
                index_map=lambda b, n, rc, vb: (b, n, NUM_BLOCKS_S - 1 - rc, vb),
            ),
            pl.BlockSpec(
                block_shape=(None, None, K, BLOCK_SIZE_V),
                index_map=lambda b, n, rc, vb: (b, n * NUM_BLOCKS_S + (NUM_BLOCKS_S - 1 - rc), 0, vb),
            ),
            h_running_spec,
        ],
        out_specs=(
            pl.BlockSpec(
                block_shape=(None, None, BLOCK_SIZE_S, K),
                index_map=lambda b, n, rc, vb: (b, n, NUM_BLOCKS_S - 1 - rc, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, None, BLOCK_SIZE_S, K),
                index_map=lambda b, n, rc, vb: (b, n, NUM_BLOCKS_S - 1 - rc, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, None, BLOCK_SIZE_S, BLOCK_SIZE_V),
                index_map=lambda b, n, rc, vb: (b, n, NUM_BLOCKS_S - 1 - rc, vb),
            ),
            h_running_spec,
        ),
        scratch_shapes=[
            pltpu.VMEM((BLOCK_SIZE_S, BLOCK_SIZE_S), jnp.float32),
            pltpu.VMEM((BLOCK_SIZE_S, K), jnp.float32),
            pltpu.VMEM((BLOCK_SIZE_S, K), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary", "arbitrary")),
    )

    return kernel(q, k, v, dy, h_checkpoints, dh)


@partial(jax.jit, static_argnames=("attention_multiplier", "BLOCK_SIZE_S", "BLOCK_SIZE_V"))
def _linear_attention_backward_pallas(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    dy: jax.Array,
    h0: jax.Array | None,
    dh: jax.Array | None,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    B, S, Nq, K = q.shape
    Nk = k.shape[-2]
    Nv, V = v.shape[-2:]

    N = max(Nq, Nk, Nv)

    Gq = N // Nq
    Gk = N // Nk
    Gv = N // Nv

    if h0 is None:
        h0 = jnp.zeros((B, N, K, V), dtype=jnp.float32)
    else:
        h0 = h0.astype(jnp.float32)

    if dh is None:
        dh = jnp.zeros((B, N, K, V), dtype=jnp.float32)
    else:
        dh = dh.astype(jnp.float32)

    q = jnp.swapaxes(q, 1, 2)
    k = jnp.swapaxes(k, 1, 2)
    v = jnp.swapaxes(v, 1, 2)
    dy = jnp.swapaxes(dy, 1, 2)

    h_checkpoints, _ = _linear_attention_checkpoint_core(
        k, v, h0, BLOCK_SIZE_S=BLOCK_SIZE_S, BLOCK_SIZE_V=BLOCK_SIZE_V
    )

    dq, dk, dv, dh0 = _linear_attention_backward_core(
        q,
        k,
        v,
        dy,
        h_checkpoints,
        dh,
        attention_multiplier=attention_multiplier,
        BLOCK_SIZE_S=BLOCK_SIZE_S,
        BLOCK_SIZE_V=BLOCK_SIZE_V,
    )

    dq = jnp.swapaxes(dq, 1, 2)
    dk = jnp.swapaxes(dk, 1, 2)
    dv = jnp.swapaxes(dv, 1, 2)

    dq = dq.reshape(B, S, Nq, Gq, K).sum(axis=3)
    dk = dk.reshape(B, S, Nk, Gk, K).sum(axis=3)
    dv = dv.reshape(B, S, Nv, Gv, V).sum(axis=3)

    return dq, dk, dv, dh0
