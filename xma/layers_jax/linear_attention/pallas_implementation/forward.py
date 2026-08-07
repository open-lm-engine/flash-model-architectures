# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from functools import partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from ....math import ceil_divide


def _state_update(h: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    h += jax.lax.dot_general(k, v, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)
    return h


def _output_readout(
    h: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    BLOCK_SIZE_S: int,
    attention_multiplier: float,
) -> jax.Array:
    dtype = q.dtype

    row = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, BLOCK_SIZE_S), 0)
    col = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, BLOCK_SIZE_S), 1)
    causal_mask = row >= col

    qk = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
    qk = jnp.where(causal_mask, qk, 0).astype(dtype)

    y = jnp.dot(qk, v, preferred_element_type=jnp.float32)
    y += jnp.dot(q, h.astype(dtype), preferred_element_type=jnp.float32)
    y *= attention_multiplier

    return y.astype(dtype)


def _forward_kernel(
    q_ref, k_ref, v_ref, h0_ref, y_ref, h_ref, *, attention_multiplier: float, BLOCK_SIZE_S: int, S: int
) -> None:
    BLOCK_ID_S = pl.program_id(3)

    @pl.when(BLOCK_ID_S == 0)
    def _():
        if h0_ref is None:
            h_ref[...] = jnp.zeros_like(h_ref)
        else:
            h_ref[...] = h0_ref[...].astype(jnp.float32)

    dtype = q_ref.dtype

    BLOCK_ID_S = pl.program_id(3)
    BLOCK_S = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, 1), 0)
    MASK_S = (BLOCK_ID_S * BLOCK_SIZE_S + BLOCK_S) < S

    q = jnp.where(MASK_S, q_ref[...], 0).astype(dtype)
    k = jnp.where(MASK_S, k_ref[...], 0).astype(dtype)
    v = jnp.where(MASK_S, v_ref[...], 0).astype(dtype)
    h = h_ref[...]

    y_ref[...] = _output_readout(
        h=h, q=q, k=k, v=v, BLOCK_SIZE_S=BLOCK_SIZE_S, attention_multiplier=attention_multiplier
    )

    h = _state_update(h=h, k=k, v=v)
    h_ref[...] = h


@partial(jax.jit, static_argnames=("attention_multiplier", "BLOCK_SIZE_S", "BLOCK_SIZE_V"))
def _forward_core(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    h0: jax.Array | None,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
) -> tuple[jax.Array, jax.Array]:
    B, Nq, S, K = q.shape
    Nk = k.shape[1]
    Nv = v.shape[1]
    V = v.shape[-1]
    N = max(Nq, Nk, Nv)

    Gq = N // Nq
    Gk = N // Nk
    Gv = N // Nv

    h_spec = pl.BlockSpec(block_shape=(None, None, K, BLOCK_SIZE_V), index_map=lambda b, n, vb, c: (b, n, 0, vb))

    kernel = pl.pallas_call(
        partial(_forward_kernel, attention_multiplier=attention_multiplier, BLOCK_SIZE_S=BLOCK_SIZE_S, S=S),
        out_shape=(
            jax.ShapeDtypeStruct(shape=(B, N, S, V), dtype=q.dtype),
            jax.ShapeDtypeStruct(shape=(B, N, K, V), dtype=jnp.float32),
        ),
        grid=(B, N, ceil_divide(V, BLOCK_SIZE_V), ceil_divide(S, BLOCK_SIZE_S)),
        in_specs=(
            pl.BlockSpec(
                block_shape=(None, None, BLOCK_SIZE_S, K),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_N, BLOCK_ID_V, BLOCK_ID_S: (
                    BLOCK_ID_B,
                    BLOCK_ID_N // Gq,
                    BLOCK_ID_S,
                    0,
                ),
            ),
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
            None if h0 is None else h_spec,
        ),
        out_specs=(
            pl.BlockSpec(
                block_shape=(None, None, BLOCK_SIZE_S, BLOCK_SIZE_V),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_N, BLOCK_ID_V, BLOCK_ID_S: (
                    BLOCK_ID_B,
                    BLOCK_ID_N,
                    BLOCK_ID_S,
                    BLOCK_ID_V,
                ),
            ),
            h_spec,
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "parallel", "arbitrary")),
    )

    y, ht = kernel(q, k, v, h0)

    return y, ht
