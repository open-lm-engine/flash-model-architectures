# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from functools import partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from ....math import ceil_divide


def _linear_attention_output_forward_kernel(
    q_ref,
    k_ref,
    v_ref,
    h_ref,
    y_ref,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
    N: int,
    Gq: int,
    Gk: int,
    Gv: int,
    NUM_BLOCKS_V: int,
) -> None:
    dtype = q_ref.dtype

    q_ = q_ref[...].transpose(1, 0, 2)
    k_ = k_ref[...].transpose(1, 0, 2)
    v_ = v_ref[...].transpose(1, 0, 2)
    h_ = h_ref[...].astype(dtype)

    row = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, BLOCK_SIZE_S), 0)
    col = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, BLOCK_SIZE_S), 1)
    causal_mask = row >= col

    for n in range(N):
        q = q_[n // Gq].astype(dtype)
        k = k_[n // Gk].astype(dtype)

        qk = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
        qk = jnp.where(causal_mask, qk, 0).astype(dtype)

        for BLOCK_ID_V in range(NUM_BLOCKS_V):
            start = BLOCK_ID_V * BLOCK_SIZE_V
            end = start + BLOCK_SIZE_V

            v = v_[n // Gv][:, start:end].astype(dtype)
            h = h_[n][:, start:end]

            y = jnp.dot(qk, v, preferred_element_type=jnp.float32)
            y += jnp.dot(q, h, preferred_element_type=jnp.float32)
            y *= attention_multiplier

            y_ref[:, n, start:end] = y.astype(y_ref.dtype)


@partial(jax.jit, static_argnames=("attention_multiplier", "BLOCK_SIZE_S", "BLOCK_SIZE_V"))
def _linear_attention_output_forward_core(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    h: jax.Array,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
) -> jax.Array:
    """computes `y` given the per-block prefix states `h` already produced by
    `_linear_attention_state_passing_core`; unlike the fused forward kernel this needs no running
    scratch state, so every (batch, sequence-block) grid cell is independent (fully "parallel"
    dimension semantics), trading the state-passing kernel's cheap sequential pass for a forward
    kernel with a smaller peak VMEM footprint than the single fused kernel.
    """

    B, S, Nq, K = q.shape
    Nk = k.shape[2]
    Nv, V = v.shape[-2:]
    N = max(Nq, Nk, Nv)

    Gq = N // Nq
    Gk = N // Nk
    Gv = N // Nv

    assert S % BLOCK_SIZE_S == 0

    NUM_BLOCKS_S = ceil_divide(S, BLOCK_SIZE_S)
    NUM_BLOCKS_V = ceil_divide(V, BLOCK_SIZE_V)
    assert (
        V == NUM_BLOCKS_V * BLOCK_SIZE_V
    ), "V must be an integer multiple of BLOCK_SIZE_V (host padding guarantees this)"

    kernel = pl.pallas_call(
        partial(
            _linear_attention_output_forward_kernel,
            attention_multiplier=attention_multiplier,
            BLOCK_SIZE_S=BLOCK_SIZE_S,
            BLOCK_SIZE_V=BLOCK_SIZE_V,
            N=N,
            Gq=Gq,
            Gk=Gk,
            Gv=Gv,
            NUM_BLOCKS_V=NUM_BLOCKS_V,
        ),
        out_shape=jax.ShapeDtypeStruct(shape=(B, S, N, V), dtype=q.dtype),
        grid=(B, NUM_BLOCKS_S),
        in_specs=(
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, Nq, K),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, BLOCK_ID_S, 0, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, Nk, K),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, BLOCK_ID_S, 0, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, Nv, V),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, BLOCK_ID_S, 0, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, N, K, V),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, BLOCK_ID_S, 0, 0),
            ),
        ),
        out_specs=pl.BlockSpec(
            block_shape=(None, BLOCK_SIZE_S, N, V),
            index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, BLOCK_ID_S, 0, 0),
        ),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True, dimension_semantics=("parallel", "parallel")),
    )

    return kernel(q, k, v, h)
