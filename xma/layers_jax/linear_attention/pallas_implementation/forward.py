# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from functools import partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from ....math import ceil_divide


def _linear_attention_forward_kernel(
    q_ref,
    k_ref,
    v_ref,
    h0_ref,
    y_ref,
    ht_ref,
    h_scratch,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
    N: int,
    Gq: int,
    Gk: int,
    Gv: int,
) -> None:
    @pl.when(pl.program_id(2) == 0)
    def _():
        if h0_ref is None:
            h_scratch[...] = jnp.zeros_like(h_scratch)
        else:
            h_scratch[...] = h0_ref[...].astype(jnp.float32)

    dtype = q_ref.dtype

    q_ = q_ref[...].transpose(1, 0, 2)
    k_ = k_ref[...].transpose(1, 0, 2)
    v_ = v_ref[...].transpose(1, 0, 2)

    row = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, BLOCK_SIZE_S), 0)
    col = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, BLOCK_SIZE_S), 1)
    causal_mask = row >= col

    for n in range(N):
        q = q_[n // Gq].astype(dtype)
        k = k_[n // Gk].astype(dtype)
        v = v_[n // Gv].astype(dtype)
        h = h_scratch[n]

        qk = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
        qk = jnp.where(causal_mask, qk, 0).astype(dtype)

        y = jnp.dot(qk, v, preferred_element_type=jnp.float32)
        y += jnp.dot(q, h.astype(dtype), preferred_element_type=jnp.float32)
        y *= attention_multiplier

        y_ref[:, n, :] = y.astype(y_ref.dtype)

        h += jax.lax.dot_general(k, v, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)
        h_scratch[n] = h

        if ht_ref is not None:
            ht_ref[n] = h.astype(ht_ref.dtype)


@partial(jax.jit, static_argnames=("attention_multiplier", "BLOCK_SIZE_S", "BLOCK_SIZE_V", "output_state"))
def _linear_attention_forward_core(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    h0: jax.Array | None,
    attention_multiplier: float,
    output_state: bool,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
) -> tuple[jax.Array, jax.Array | None]:
    B, S, Nq, K = q.shape
    Nk = k.shape[2]
    Nv = v.shape[2]
    V = v.shape[-1]

    N = max(Nq, Nk, Nv)
    Gq = N // Nq
    Gk = N // Nk
    Gv = N // Nv

    assert S % BLOCK_SIZE_S == 0
    assert V % BLOCK_SIZE_V == 0

    h_spec = pl.BlockSpec(
        block_shape=(None, N, K, BLOCK_SIZE_V),
        index_map=lambda BLOCK_ID_B, BLOCK_ID_V, _: (BLOCK_ID_B, 0, 0, BLOCK_ID_V),
    )

    kernel = pl.pallas_call(
        partial(
            _linear_attention_forward_kernel,
            attention_multiplier=attention_multiplier,
            BLOCK_SIZE_S=BLOCK_SIZE_S,
            N=N,
            Gq=Gq,
            Gk=Gk,
            Gv=Gv,
        ),
        out_shape=(
            jax.ShapeDtypeStruct(shape=(B, S, N, V), dtype=q.dtype),
            jax.ShapeDtypeStruct(shape=(B, N, K, V), dtype=jnp.float32) if output_state else None,
        ),
        grid=(B, ceil_divide(V, BLOCK_SIZE_V), ceil_divide(S, BLOCK_SIZE_S)),
        in_specs=(
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, Nq, K),
                index_map=lambda BLOCK_ID_B, _, BLOCK_ID_S: (BLOCK_ID_B, BLOCK_ID_S, 0, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, Nk, K),
                index_map=lambda BLOCK_ID_B, _, BLOCK_ID_S: (BLOCK_ID_B, BLOCK_ID_S, 0, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, Nv, BLOCK_SIZE_V),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_V, BLOCK_ID_S: (BLOCK_ID_B, BLOCK_ID_S, 0, BLOCK_ID_V),
            ),
            None if h0 is None else h_spec,
        ),
        out_specs=(
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, N, BLOCK_SIZE_V),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_V, BLOCK_ID_S: (BLOCK_ID_B, BLOCK_ID_S, 0, BLOCK_ID_V),
            ),
            h_spec if output_state else None,
        ),
        scratch_shapes=[pltpu.VMEM((N, K, BLOCK_SIZE_V), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True, dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
    )

    y, ht = kernel(q, k, v, h0)

    return y, ht
