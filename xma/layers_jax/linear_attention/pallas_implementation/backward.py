# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from functools import partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from ....math import ceil_divide
from .forward import _state_update


def _compute_state_passing(k_ref, v_ref, h_checkpoint_ref, h_ref, *, BLOCK_SIZE_S: int, S: int) -> None:
    dtype = k_ref.dtype

    BLOCK_ID_S = pl.program_id(3)
    BLOCK_S = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, 1), 0)
    MASK_S = (BLOCK_ID_S * BLOCK_SIZE_S + BLOCK_S) < S

    k = jnp.where(MASK_S, k_ref[...], 0).astype(dtype)
    v = jnp.where(MASK_S, v_ref[...], 0).astype(dtype)
    h = h_ref[...]

    h_checkpoint_ref[...] = h
    h_ref[...] = _state_update(h=h, k=k, v=v)


def _state_passing_kernel(k_ref, v_ref, h0_ref, h_checkpoint_ref, h_ref, *, BLOCK_SIZE_S: int, S: int) -> None:
    @pl.when(pl.program_id(3) == 0)
    def _():
        h_ref[...] = h0_ref[...].astype(jnp.float32)

    _compute_state_passing(
        k_ref=k_ref, v_ref=v_ref, h_checkpoint_ref=h_checkpoint_ref, h_ref=h_ref, BLOCK_SIZE_S=BLOCK_SIZE_S, S=S
    )


def _state_passing_zero_h0_kernel(k_ref, v_ref, h_checkpoint_ref, h_ref, *, BLOCK_SIZE_S: int, S: int) -> None:
    @pl.when(pl.program_id(3) == 0)
    def _():
        h_ref[...] = jnp.zeros_like(h_ref)

    _compute_state_passing(
        k_ref=k_ref, v_ref=v_ref, h_checkpoint_ref=h_checkpoint_ref, h_ref=h_ref, BLOCK_SIZE_S=BLOCK_SIZE_S, S=S
    )


@partial(jax.jit, static_argnames=("N", "BLOCK_SIZE_S", "BLOCK_SIZE_V"))
def _state_passing_core(
    k: jax.Array, v: jax.Array, h0: jax.Array | None, N: int, BLOCK_SIZE_S: int, BLOCK_SIZE_V: int
) -> tuple[jax.Array, jax.Array]:
    B, Nk, S, K = k.shape
    Nv = v.shape[1]
    V = v.shape[-1]

    Gk = N // Nk
    Gv = N // Nv

    NUM_BLOCKS_S = ceil_divide(S, BLOCK_SIZE_S)

    in_specs = [
        pl.BlockSpec(block_shape=(None, None, BLOCK_SIZE_S, K), index_map=lambda b, n, vb, c: (b, n // Gk, c, 0)),
        pl.BlockSpec(
            block_shape=(None, None, BLOCK_SIZE_S, BLOCK_SIZE_V), index_map=lambda b, n, vb, c: (b, n // Gv, c, vb)
        ),
    ]

    h_spec = pl.BlockSpec(block_shape=(None, None, K, BLOCK_SIZE_V), index_map=lambda b, n, vb, c: (b, n, 0, vb))

    if h0 is None:
        kernel_fn = _state_passing_zero_h0_kernel
        args = (k, v)
    else:
        kernel_fn = _state_passing_kernel
        in_specs += [h_spec]
        args = (k, v, h0)

    kernel = pl.pallas_call(
        partial(kernel_fn, BLOCK_SIZE_S=BLOCK_SIZE_S, S=S),
        out_shape=(
            jax.ShapeDtypeStruct(shape=(B, N * NUM_BLOCKS_S, K, V), dtype=jnp.float32),
            jax.ShapeDtypeStruct(shape=(B, N, K, V), dtype=jnp.float32),
        ),
        grid=(B, N, ceil_divide(V, BLOCK_SIZE_V), NUM_BLOCKS_S),
        in_specs=in_specs,
        out_specs=(
            pl.BlockSpec(
                block_shape=(None, None, K, BLOCK_SIZE_V),
                index_map=lambda b, n, vb, c: (b, n * NUM_BLOCKS_S + c, 0, vb),
            ),
            h_spec,
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "parallel", "arbitrary")),
    )

    return kernel(*args)


def _backward(
    q_ref,
    k_ref,
    v_ref,
    dy_ref,
    h_checkpoint_ref,
    dh0_ref,
    dq_ref,
    dk_ref,
    dv_ref,
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


def _backward_kernel(
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

    @pl.when(rc == 0)
    def _():
        dh0_ref[...] = dht_ref[...]

    _backward(
        q_ref=q_ref,
        k_ref=k_ref,
        v_ref=v_ref,
        dy_ref=dy_ref,
        h_checkpoint_ref=h_checkpoint_ref,
        dh0_ref=dh0_ref,
        dq_ref=dq_ref,
        dk_ref=dk_ref,
        dv_ref=dv_ref,
        dqk_ref=dqk_ref,
        dq_term2_ref=dq_term2_ref,
        dk_term2_ref=dk_term2_ref,
        attention_multiplier=attention_multiplier,
        BLOCK_SIZE_S=BLOCK_SIZE_S,
        S=S,
        V=V,
        NUM_BLOCKS_S=NUM_BLOCKS_S,
        NUM_V_TILES=NUM_V_TILES,
    )


def _backward_zero_dh_kernel(
    q_ref,
    k_ref,
    v_ref,
    h_checkpoint_ref,
    dy_ref,
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

    @pl.when(rc == 0)
    def _():
        dh0_ref[...] = jnp.zeros_like(dh0_ref)

    _backward(
        q_ref=q_ref,
        k_ref=k_ref,
        v_ref=v_ref,
        dy_ref=dy_ref,
        h_checkpoint_ref=h_checkpoint_ref,
        dh0_ref=dh0_ref,
        dq_ref=dq_ref,
        dk_ref=dk_ref,
        dv_ref=dv_ref,
        dqk_ref=dqk_ref,
        dq_term2_ref=dq_term2_ref,
        dk_term2_ref=dk_term2_ref,
        attention_multiplier=attention_multiplier,
        BLOCK_SIZE_S=BLOCK_SIZE_S,
        S=S,
        V=V,
        NUM_BLOCKS_S=NUM_BLOCKS_S,
        NUM_V_TILES=NUM_V_TILES,
    )


@partial(jax.jit, static_argnames=("attention_multiplier", "BLOCK_SIZE_S", "BLOCK_SIZE_V"))
def _backward_core(
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

    h_spec = pl.BlockSpec(block_shape=(None, None, K, BLOCK_SIZE_V), index_map=lambda b, n, rc, vb: (b, n, 0, vb))

    in_specs = [
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
            block_shape=(None, None, K, BLOCK_SIZE_V),
            index_map=lambda b, n, rc, vb: (b, n * NUM_BLOCKS_S + (NUM_BLOCKS_S - 1 - rc), 0, vb),
        ),
        pl.BlockSpec(
            block_shape=(None, None, BLOCK_SIZE_S, BLOCK_SIZE_V),
            index_map=lambda b, n, rc, vb: (b, n, NUM_BLOCKS_S - 1 - rc, vb),
        ),
    ]

    if dht is None:
        kernel_fn = _backward_zero_dh_kernel
        args = (q, k, v, h, dy)
    else:
        kernel_fn = _backward_kernel
        in_specs += [h_spec]
        args = (q, k, v, h, dy, dht)

    kernel = pl.pallas_call(
        partial(
            kernel_fn,
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
        in_specs=in_specs,
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
            h_spec,
        ),
        scratch_shapes=[
            pltpu.VMEM((BLOCK_SIZE_S, BLOCK_SIZE_S), jnp.float32),
            pltpu.VMEM((BLOCK_SIZE_S, K), jnp.float32),
            pltpu.VMEM((BLOCK_SIZE_S, K), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary", "arbitrary")),
    )

    return kernel(*args)


@partial(jax.jit, static_argnames=("attention_multiplier", "BLOCK_SIZE_S", "BLOCK_SIZE_V"))
def _linear_attention_backward_pallas(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    h0: jax.Array | None,
    dy: jax.Array,
    dht: jax.Array | None,
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

    q = jnp.swapaxes(q, 1, 2)
    k = jnp.swapaxes(k, 1, 2)
    v = jnp.swapaxes(v, 1, 2)
    dy = jnp.swapaxes(dy, 1, 2)

    h, _ = _state_passing_core(k=k, v=v, h0=h0, N=N, BLOCK_SIZE_S=BLOCK_SIZE_S, BLOCK_SIZE_V=BLOCK_SIZE_V)

    dq, dk, dv, dh0 = _backward_core(
        q=q,
        k=k,
        v=v,
        h=h,
        dy=dy,
        dht=dht,
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
