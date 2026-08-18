# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from functools import partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from ....math import ceil_divide
from .forward import _get_causal_mask


def _linear_attention_backward_kernel(
    q_ref,
    k_ref,
    v_ref,
    log_f_cumsum_ref,
    h_ref,
    dy_ref,
    dht_ref,
    dq_ref,
    dk_ref,
    dv_ref,
    df_ref,
    dh0_ref,
    dh_scratch,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
    N: int,
    Gq: int,
    Gk: int,
    Gv: int,
    Gf: int | None,
    f_diagonal: bool,
    NUM_BLOCKS_V: int,
    NUM_BLOCKS_S: int,
) -> None:
    S_CELLS_VISITED = pl.program_id(1)

    @pl.when(S_CELLS_VISITED == 0)
    def _():
        if dht_ref is None:
            dh_scratch[...] = jnp.zeros_like(dh_scratch)
        else:
            dh_scratch[...] = dht_ref[...].astype(jnp.float32)

    dtype = q_ref.dtype

    q_ = q_ref[...].transpose(1, 0, 2)
    k_ = k_ref[...].transpose(1, 0, 2)
    v_ = v_ref[...].transpose(1, 0, 2)
    dy_ = dy_ref[...].transpose(1, 0, 2)
    h_ = h_ref[...].astype(dtype)

    log_f_cumsum_ = None
    if log_f_cumsum_ref is not None:
        log_f_cumsum_ = log_f_cumsum_ref[...]

        if f_diagonal:
            log_f_cumsum_ = log_f_cumsum_.transpose(1, 0, 2)
        else:
            log_f_cumsum_ = log_f_cumsum_.transpose(1, 0)

    causal_mask = _get_causal_mask(BLOCK_SIZE_S)
    K = q_.shape[-1]

    for n in range(N):
        q = q_[n // Gq].astype(dtype)
        k = k_[n // Gk].astype(dtype)
        dy_full = (dy_[n].astype(jnp.float32) * attention_multiplier).astype(dtype)

        if log_f_cumsum_ref is None:
            qk = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
            qk = jnp.where(causal_mask, qk, 0).astype(dtype)

            qf = q
            kf_last = k
            kf_inv = None
        else:
            log_f = log_f_cumsum_[n // Gf].astype(jnp.float32)
            log_f_last = log_f[-1]

            f = jnp.exp(log_f)
            f_last = jnp.exp(log_f_last)

            if f_diagonal:
                f_to_last = jnp.exp(log_f_last[None, :] - log_f)
                f_inv = jnp.exp(-log_f)

                qf = (q * f).astype(dtype)
                kf_inv = (k * f_inv).astype(dtype)
                kf_last = (k * f_to_last).astype(dtype)

                qk = jax.lax.dot_general(qf, kf_inv, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
            else:
                f_to_last = jnp.exp(log_f_last - log_f)
                f_ij = jnp.exp(log_f[:, None] - log_f[None, :])

                qf = (q * f[:, None]).astype(dtype)
                kf_last = (k * f_to_last[:, None]).astype(dtype)
                kf_inv = None

                qk = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
                qk *= f_ij

            qk = jnp.where(causal_mask, qk, 0).astype(dtype)

        dyv = jnp.zeros((BLOCK_SIZE_S, BLOCK_SIZE_S), jnp.float32)
        dq = jnp.zeros((BLOCK_SIZE_S, K), jnp.float32)
        dk = jnp.zeros((BLOCK_SIZE_S, K), jnp.float32)

        if log_f_cumsum_ is not None:
            df = jnp.zeros((BLOCK_SIZE_S, K) if f_diagonal else (BLOCK_SIZE_S,), jnp.float32)
            df_last = jnp.zeros((K,) if f_diagonal else (), jnp.float32)

        for BLOCK_ID_V in range(NUM_BLOCKS_V):
            start = BLOCK_ID_V * BLOCK_SIZE_V
            end = start + BLOCK_SIZE_V

            v = v_[n // Gv][:, start:end].astype(dtype)
            dy = dy_full[:, start:end]
            h = h_[n][:, start:end]
            dh = dh_scratch[n][:, start:end]

            dv = jax.lax.dot_general(qk, dy, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)
            dv += jax.lax.dot_general(
                kf_last, dh.astype(dtype), (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
            )
            dv_ref[:, n, start:end] = dv.astype(dtype)

            _dq = jax.lax.dot_general(dy, h, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
            if log_f_cumsum_ is not None:
                _dq *= f if f_diagonal else f[:, None]
            dq += _dq

            _dk = jax.lax.dot_general(v, dh, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
            if log_f_cumsum_ is not None:
                _dk *= f_to_last if f_diagonal else f_to_last[:, None]
            dk += _dk

            if log_f_cumsum_ is not None:
                df_q = q.astype(jnp.float32) * _dq
                df_k = k.astype(jnp.float32) * _dk
                df_decay = h.astype(jnp.float32) * dh

                if f_diagonal:
                    df += df_q - df_k
                    df_last += df_k.sum(axis=0) + f_last * df_decay.sum(axis=1)
                else:
                    df += df_q.sum(axis=-1) - df_k.sum(axis=-1)
                    df_last += df_k.sum() + f_last * df_decay.sum()

            if log_f_cumsum_ is not None:
                dh *= f_last[:, None] if f_diagonal else f_last
            dh += jax.lax.dot_general(qf, dy, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)
            dh_scratch[n, :, start:end] = dh

            dyv += jax.lax.dot_general(dy, v, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)

        dyv = jnp.where(causal_mask, dyv, 0)
        if log_f_cumsum_ is not None and not f_diagonal:
            dyv *= f_ij
        dyv = dyv.astype(dtype)

        dq_intra = jax.lax.dot_general(
            dyv, k if kf_inv is None else kf_inv, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
        )

        dk_intra = jax.lax.dot_general(
            dyv, q if kf_inv is None else qf, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32
        )

        if f_diagonal:
            dq_intra *= f
            dk_intra *= f_inv

        dq += dq_intra
        dk += dk_intra

        dq_ref[:, n, :] = dq.astype(dtype)
        dk_ref[:, n, :] = dk.astype(dtype)

        if log_f_cumsum_ is not None:
            suffix_sum = jnp.where(_get_causal_mask(BLOCK_SIZE_S, transpose=True), 1.0, 0.0)

            if f_diagonal:
                df += q.astype(jnp.float32) * dq_intra - k.astype(jnp.float32) * dk_intra
                dlog_f = jax.lax.dot_general(suffix_sum, df, (((1,), (0,)), ((), ()))) + df_last
                df_ref[:, n, :] = dlog_f.astype(df_ref.dtype)
            else:
                df += (q.astype(jnp.float32) * dq_intra).sum(axis=-1)
                df -= (k.astype(jnp.float32) * dk_intra).sum(axis=-1)
                dlog_f = jax.lax.dot_general(suffix_sum, df, (((1,), (0,)), ((), ()))) + df_last
                df_ref[:, n] = dlog_f.astype(df_ref.dtype)

    @pl.when(S_CELLS_VISITED == NUM_BLOCKS_S - 1)
    def _():
        dh0_ref[...] = dh_scratch[...]


@partial(jax.jit, static_argnames=("attention_multiplier", "BLOCK_SIZE_S", "BLOCK_SIZE_V"))
def _linear_attention_backward_core(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_f_cumsum: jax.Array | None,
    h: jax.Array,
    dy: jax.Array,
    dht: jax.Array | None,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    B, S, Nq, K = q.shape
    Nk = k.shape[2]
    Nv, V = v.shape[-2:]
    Nf = 0 if log_f_cumsum is None else log_f_cumsum.shape[2]
    N = dy.shape[2]

    Gq = N // Nq
    Gk = N // Nk
    Gv = N // Nv
    Gf = None if log_f_cumsum is None else N // Nf

    f_diagonal = log_f_cumsum is not None and log_f_cumsum.ndim == 4
    if log_f_cumsum is not None:
        assert log_f_cumsum.shape == (B, S, Nf, K) if f_diagonal else (B, S, Nf)

    assert S % BLOCK_SIZE_S == 0

    NUM_BLOCKS_S = ceil_divide(S, BLOCK_SIZE_S)
    NUM_BLOCKS_V = ceil_divide(V, BLOCK_SIZE_V)
    assert (
        V == NUM_BLOCKS_V * BLOCK_SIZE_V
    ), "V must be an integer multiple of BLOCK_SIZE_V (host padding guarantees this)"

    f_spec = None
    df_spec = None
    if log_f_cumsum is not None:
        if f_diagonal:
            f_spec = pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, Nf, K),
                index_map=lambda B, S: (B, NUM_BLOCKS_S - 1 - S, 0, 0),
            )

            df_spec = pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, N, K),
                index_map=lambda B, S: (B, NUM_BLOCKS_S - 1 - S, 0, 0),
            )
        else:
            f_spec = pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, Nf),
                index_map=lambda B, S: (B, NUM_BLOCKS_S - 1 - S, 0),
            )

            df_spec = pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, N),
                index_map=lambda B, S: (B, NUM_BLOCKS_S - 1 - S, 0),
            )

    kernel = pl.pallas_call(
        partial(
            _linear_attention_backward_kernel,
            attention_multiplier=attention_multiplier,
            BLOCK_SIZE_S=BLOCK_SIZE_S,
            BLOCK_SIZE_V=BLOCK_SIZE_V,
            N=N,
            Gq=Gq,
            Gk=Gk,
            Gv=Gv,
            Gf=Gf,
            f_diagonal=f_diagonal,
            NUM_BLOCKS_V=NUM_BLOCKS_V,
            NUM_BLOCKS_S=NUM_BLOCKS_S,
        ),
        out_shape=(
            jax.ShapeDtypeStruct(shape=(B, S, N, K), dtype=q.dtype),
            jax.ShapeDtypeStruct(shape=(B, S, N, K), dtype=q.dtype),
            jax.ShapeDtypeStruct(shape=(B, S, N, V), dtype=q.dtype),
            (
                None
                if log_f_cumsum is None
                else jax.ShapeDtypeStruct(shape=(B, S, N, K) if f_diagonal else (B, S, N), dtype=jnp.float32)
            ),
            jax.ShapeDtypeStruct(shape=(B, N, K, V), dtype=jnp.float32),
        ),
        grid=(B, NUM_BLOCKS_S),
        in_specs=(
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, Nq, K),
                index_map=lambda B, S: (B, NUM_BLOCKS_S - 1 - S, 0, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, Nk, K),
                index_map=lambda B, S: (B, NUM_BLOCKS_S - 1 - S, 0, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, Nv, V),
                index_map=lambda B, S: (B, NUM_BLOCKS_S - 1 - S, 0, 0),
            ),
            f_spec,
            pl.BlockSpec(
                block_shape=(None, N, K, V),
                index_map=lambda B, S: (B, NUM_BLOCKS_S - 1 - S, 0, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, N, V),
                index_map=lambda B, S: (B, NUM_BLOCKS_S - 1 - S, 0, 0),
            ),
            (
                None
                if dht is None
                else pl.BlockSpec(
                    block_shape=(None, N, K, V),
                    index_map=lambda B, _: (B, 0, 0, 0),
                )
            ),
        ),
        out_specs=(
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, N, K),
                index_map=lambda B, S: (B, NUM_BLOCKS_S - 1 - S, 0, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, N, K),
                index_map=lambda B, S: (B, NUM_BLOCKS_S - 1 - S, 0, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, N, V),
                index_map=lambda B, S: (B, NUM_BLOCKS_S - 1 - S, 0, 0),
            ),
            df_spec,
            pl.BlockSpec(
                block_shape=(None, N, K, V),
                index_map=lambda B, S: (B, 0, 0, 0),
            ),
        ),
        scratch_shapes=[pltpu.VMEM((N, K, V), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True, dimension_semantics=("parallel", "arbitrary")
        ),
    )

    return kernel(q, k, v, log_f_cumsum, h, dy, dht)
