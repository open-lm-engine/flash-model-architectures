# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from functools import partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from ....math import ceil_divide


def _get_causal_mask(BLOCK_SIZE_S):
    row = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, BLOCK_SIZE_S), 0)
    col = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, BLOCK_SIZE_S), 1)
    causal_mask = row >= col
    return causal_mask


def _linear_attention_forward_kernel(
    q_ref,
    k_ref,
    v_ref,
    log_f_cumsum_ref,
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
    Gf: int | None,
    f_diagonal: bool,
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

    log_f_cumsum_ = None
    if log_f_cumsum_ref is not None:
        log_f_cumsum_ = log_f_cumsum_ref[...]

        if f_diagonal:
            log_f_cumsum_ = log_f_cumsum_.transpose(1, 0, 2)
        else:
            log_f_cumsum_ = log_f_cumsum_.transpose(1, 0)

    causal_mask = _get_causal_mask(BLOCK_SIZE_S)

    for n in range(N):
        q = q_[n // Gq].astype(dtype)
        k = k_[n // Gk].astype(dtype)
        v = v_[n // Gv].astype(dtype)
        h = h_scratch[n]

        if log_f_cumsum_ref is None:
            qk = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
            qk = jnp.where(causal_mask, qk, 0).astype(dtype)

            y = jnp.dot(qk, v, preferred_element_type=jnp.float32)
            y += jnp.dot(q, h.astype(dtype), preferred_element_type=jnp.float32)
        else:
            log_f = log_f_cumsum_[n // Gf].astype(jnp.float32)
            log_f_last = log_f[-1]

            if f_diagonal:
                qf = (q * jnp.exp(log_f)).astype(dtype)
                kf_inv = (k * jnp.exp(-log_f)).astype(dtype)

                qk = jax.lax.dot_general(qf, kf_inv, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
            else:
                qf = (q * jnp.exp(log_f[:, None])).astype(dtype)

                qk = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
                qk *= jnp.exp(log_f[:, None] - log_f[None, :])

            qk = jnp.where(causal_mask, qk, 0).astype(dtype)
            y = jnp.dot(qk, v, preferred_element_type=jnp.float32)
            y += jnp.dot(qf, h.astype(dtype), preferred_element_type=jnp.float32)

            if f_diagonal:
                h *= jnp.exp(log_f_last[:, None])
                k *= jnp.exp(log_f_last[None, :] - log_f)
            else:
                h *= jnp.exp(log_f_last)
                k *= jnp.exp(log_f_last - log_f)[:, None]

        k = k.astype(dtype)

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
    log_f_cumsum: jax.Array | None,
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

    Nf = 0 if log_f_cumsum is None else log_f_cumsum.shape[2]
    f_diagonal = log_f_cumsum is not None and log_f_cumsum.ndim == 4

    N = max(Nq, Nk, Nv, Nf)
    Gq = N // Nq
    Gk = N // Nk
    Gv = N // Nv
    Gf = None if log_f_cumsum is None else N // Nf

    assert S % BLOCK_SIZE_S == 0
    assert V % BLOCK_SIZE_V == 0

    h_spec = pl.BlockSpec(
        block_shape=(None, N, K, BLOCK_SIZE_V),
        index_map=lambda BLOCK_ID_B, BLOCK_ID_V, _: (BLOCK_ID_B, 0, 0, BLOCK_ID_V),
    )

    f_spec = None
    if log_f_cumsum is not None:
        if f_diagonal:
            f_spec = pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, Nf, K),
                index_map=lambda BLOCK_ID_B, _, BLOCK_ID_S: (BLOCK_ID_B, BLOCK_ID_S, 0, 0),
            )
        else:
            f_spec = pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, Nf),
                index_map=lambda BLOCK_ID_B, _, BLOCK_ID_S: (BLOCK_ID_B, BLOCK_ID_S, 0),
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
            Gf=Gf,
            f_diagonal=f_diagonal,
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
            f_spec,
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

    y, ht = kernel(q, k, v, log_f_cumsum, h0)

    return y, ht
