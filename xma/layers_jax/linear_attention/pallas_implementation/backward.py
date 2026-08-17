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
    f_cumsum_ref,
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
    hc_ = h_ref[...].astype(dtype)

    f_cumsum_ = None
    if f_cumsum_ref is not None:
        f_cumsum_ = f_cumsum_ref[...]

        if f_diagonal:
            f_cumsum_ = f_cumsum_.transpose(1, 0, 2)
        else:
            f_cumsum_ = f_cumsum_.transpose(1, 0)

    causal_mask = _get_causal_mask(BLOCK_SIZE_S)
    K = q_.shape[-1]

    for n in range(N):
        q = q_[n // Gq].astype(dtype)
        k = k_[n // Gk].astype(dtype)
        dy_full = (dy_[n].astype(jnp.float32) * attention_multiplier).astype(dtype)

        if f_cumsum_ is None:
            qk = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
            qk = jnp.where(causal_mask, qk, 0).astype(dtype)

            q_inter = q
            k_state = k
        elif f_diagonal:
            # diagonal (per-key-dimension) gate: as in the forward, the intra-chunk decay is folded
            # into the matmul operands since it does not factor out of the q@k contraction;
            # exp(c), exp(c_last) and exp(c_last - c) are bounded in [0, 1], while exp(-c) grows with
            # the in-chunk decay budget and is finite as long as |c_last| < ~88 (fp32 range)
            c = f_cumsum_[n // Gf]
            c_last = c[-1]
            e_c = jnp.exp(c)
            e_last = jnp.exp(c_last)
            e_w2 = jnp.exp(c_last[None, :] - c)

            q_inter = (q * e_c).astype(dtype)
            k_inter = (k * jnp.exp(-c)).astype(dtype)
            k_state = (k * e_w2).astype(dtype)

            A = jax.lax.dot_general(q_inter, k_inter, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
            A = jnp.where(causal_mask, A, 0).astype(dtype)
            D = None
        else:
            # scalar (per-position) gate: f_cumsum is the chunk-local inclusive cumsum of the
            # per-position log-decay, so every exponent is non-positive and exp() stays in [0, 1]
            c = f_cumsum_[n // Gf]
            c_last = c[-1]
            e_c = jnp.exp(c)
            e_last = jnp.exp(c_last)
            e_w2 = jnp.exp(c_last - c)
            D = jnp.exp(c[:, None] - c[None, :])

            A = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
            A = jnp.where(causal_mask, A * D, 0).astype(dtype)

            q_inter = (q * e_c[:, None]).astype(dtype)
            k_state = (k * e_w2[:, None]).astype(dtype)
            k_inter = None

        dyv = jnp.zeros((BLOCK_SIZE_S, BLOCK_SIZE_S), jnp.float32)
        dq = jnp.zeros((BLOCK_SIZE_S, K), jnp.float32)
        dk = jnp.zeros((BLOCK_SIZE_S, K), jnp.float32)

        if f_cumsum_ is not None:
            # dc: gradient wrt the chunk-local cumsum c. Every c-dependent path satisfies the
            # identity dc = q * dq(q-side terms) - k * dk(k-side terms) except the state decay
            # term exp(c_last) * h, which contributes directly to c[-1] (accumulated in dcL)
            dc = jnp.zeros((BLOCK_SIZE_S, K) if f_diagonal else (BLOCK_SIZE_S,), jnp.float32)
            dcL = jnp.zeros((K,) if f_diagonal else (), jnp.float32)

        for BLOCK_ID_V in range(NUM_BLOCKS_V):
            start = BLOCK_ID_V * BLOCK_SIZE_V
            end = start + BLOCK_SIZE_V

            v = v_[n // Gv][:, start:end].astype(dtype)
            dy = dy_full[:, start:end]
            hc = hc_[n][:, start:end]
            dh = dh_scratch[n][:, start:end]

            dv = jax.lax.dot_general(qk, dy, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)
            dv += jax.lax.dot_general(
                k_state, dh.astype(dtype), (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
            )
            dv_ref[:, n, start:end] = dv.astype(dtype)

            dh_next = dh
            if f_cumsum_ is not None:
                dh_next = dh * (e_last[:, None] if f_diagonal else e_last)
            dh_next += jax.lax.dot_general(q_inter, dy, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)
            dh_scratch[n, :, start:end] = dh_next

            dyv += jax.lax.dot_general(dy, v, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)

            dq_inter = jax.lax.dot_general(dy, hc, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
            if f_cumsum_ is not None:
                dq_inter = dq_inter * (e_c if f_diagonal else e_c[:, None])
            dq += dq_inter

            dk_state = jax.lax.dot_general(v, dh, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
            if f_cumsum_ is not None:
                dk_state = dk_state * (e_w2 if f_diagonal else e_w2[:, None])
            dk += dk_state

            if f_cumsum_ is not None:
                dc_q = q.astype(jnp.float32) * dq_inter
                dc_k = k.astype(jnp.float32) * dk_state
                # state decay term exp(c_last) * h contributes to c[-1]
                dc_decay = hc.astype(jnp.float32) * dh

                if f_diagonal:
                    dc += dc_q - dc_k
                    dcL += dc_k.sum(axis=0) + e_last * dc_decay.sum(axis=1)
                else:
                    dc += dc_q.sum(axis=-1) - dc_k.sum(axis=-1)
                    dcL += dc_k.sum() + e_last * dc_decay.sum()

        dyv = jnp.where(causal_mask, dyv, 0)
        if f_cumsum_ is not None and not f_diagonal:
            # scalar gate: the intra-chunk decay exp(c[i] - c[j]) sits on the masked (S, S) matrix
            dyv = dyv * D
        dyv = dyv.astype(dtype)

        dq_intra = jax.lax.dot_general(
            dyv, k if k_inter is None else k_inter, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
        )

        dk_intra = jax.lax.dot_general(
            dyv, q if k_inter is None else q_inter, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32
        )

        if f_diagonal:
            dq_intra = dq_intra * e_c
            dk_intra = dk_intra * jnp.exp(-c)

        dq += dq_intra
        dk += dk_intra

        dq_ref[:, n, :] = dq.astype(dtype)
        dk_ref[:, n, :] = dk.astype(dtype)

        if f_cumsum_ is not None:
            if f_diagonal:
                dc += q.astype(jnp.float32) * dq_intra - k.astype(jnp.float32) * dk_intra
                df_ref[:, n, :] = dc.at[-1, :].add(dcL).astype(df_ref.dtype)
            else:
                dc += (q.astype(jnp.float32) * dq_intra).sum(axis=-1)
                dc -= (k.astype(jnp.float32) * dk_intra).sum(axis=-1)
                df_ref[:, n] = dc.at[-1].add(dcL).astype(df_ref.dtype)

    @pl.when(S_CELLS_VISITED == NUM_BLOCKS_S - 1)
    def _():
        dh0_ref[...] = dh_scratch[...]


@partial(jax.jit, static_argnames=("attention_multiplier", "BLOCK_SIZE_S", "BLOCK_SIZE_V"))
def _linear_attention_backward_core(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    f_cumsum: jax.Array | None,
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
    Nf = 0 if f_cumsum is None else f_cumsum.shape[2]
    N = dy.shape[2]

    Gq = N // Nq
    Gk = N // Nk
    Gv = N // Nv
    Gf = None if f_cumsum is None else N // Nf

    f_diagonal = f_cumsum is not None and f_cumsum.ndim == 4
    if f_cumsum is not None:
        assert f_cumsum.shape == (B, S, Nf, K) if f_diagonal else (B, S, Nf)

    assert S % BLOCK_SIZE_S == 0

    NUM_BLOCKS_S = ceil_divide(S, BLOCK_SIZE_S)
    NUM_BLOCKS_V = ceil_divide(V, BLOCK_SIZE_V)
    assert (
        V == NUM_BLOCKS_V * BLOCK_SIZE_V
    ), "V must be an integer multiple of BLOCK_SIZE_V (host padding guarantees this)"

    f_spec = None
    df_spec = None
    if f_cumsum is not None:
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
                if f_cumsum is None
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

    return kernel(q, k, v, f_cumsum, h, dy, dht)
