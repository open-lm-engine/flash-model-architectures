# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from functools import partial

import jax
import jax.numpy as jnp

from ....math import ceil_divide
from .backward import _linear_attention_backward_core
from .forward import _BATCHED_HEAD_GROUP, _linear_attention_forward_core
from .state_passing import _linear_attention_state_passing_core


_MAX_HEADS_PER_PALLAS_CELL = 16


def _cumulative_log_decay(log_f: jax.Array | None, BLOCK_SIZE_S: int) -> jax.Array | None:
    if log_f is None:
        return None

    B, S = log_f.shape[:2]
    assert S % BLOCK_SIZE_S == 0

    log_f = log_f.reshape(B, S // BLOCK_SIZE_S, BLOCK_SIZE_S, *log_f.shape[2:])
    log_f = log_f.astype(jnp.float32)
    log_f_cumsum = jnp.cumsum(log_f, axis=2)
    log_f_cumsum = log_f_cumsum.reshape(B, S, *log_f.shape[3:])

    return log_f_cumsum


@partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8))
def _linear_attention_pallas(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_f: jax.Array | None,
    h0: jax.Array | None,
    attention_multiplier: float,
    output_state: bool,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
) -> tuple[jax.Array, jax.Array]:
    # diagonal gates on the batched fast path skip the host-side chunk-local cumsum
    # entirely: the batched kernel scans raw log_f inside pallas via a triangular
    # systolic matmul. The host jnp.cumsum costs 1.8 ms at production (268 MB fp32
    # scan) -- far more than the whole fused kernel (0.70 ms). This is only safe when
    # the batched path is guaranteed to trigger (identical eligibility condition to
    # forward.py's `batched` flag).
    fused_diag_scan = False
    if log_f is not None and log_f.ndim == 4:
        Nq, Nk, Nv = q.shape[2], k.shape[2], v.shape[2]
        N = max(Nq, Nk, Nv, log_f.shape[2])
        fused_diag_scan = Nq == N and Nk == N and Nv == N and log_f.shape[2] == N and N % _BATCHED_HEAD_GROUP == 0

    # fused diagonal feeds RAW log_f (any dtype) to the kernel: forward_kernel upcasts each
    # loaded tile to fp32 in-VMEM before the triangular scan matmul, so a host-side astype
    # would only burn a 268 MB read+write pass (~0.34 ms at production shape).
    log_f_cumsum = log_f if fused_diag_scan else _cumulative_log_decay(log_f, BLOCK_SIZE_S)

    return _linear_attention_forward_core(
        q=q,
        k=k,
        v=v,
        log_f_cumsum=log_f_cumsum,
        h0=h0,
        attention_multiplier=attention_multiplier,
        output_state=output_state,
        BLOCK_SIZE_S=BLOCK_SIZE_S,
        BLOCK_SIZE_V=BLOCK_SIZE_V,
        fused_diag_scan=fused_diag_scan,
    )


def _linear_attention_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_f: jax.Array | None,
    h0: jax.Array | None,
    attention_multiplier: float,
    output_state: bool,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
) -> tuple[tuple[jax.Array, jax.Array | None], tuple]:
    log_f_cumsum = _cumulative_log_decay(log_f, BLOCK_SIZE_S)

    y, h = _linear_attention_forward_core(
        q=q,
        k=k,
        v=v,
        log_f_cumsum=log_f_cumsum,
        h0=h0,
        attention_multiplier=attention_multiplier,
        output_state=output_state,
        BLOCK_SIZE_S=BLOCK_SIZE_S,
        BLOCK_SIZE_V=BLOCK_SIZE_V,
    )

    return (y, h), (q, k, v, log_f_cumsum, h0)


@partial(jax.jit, static_argnames=("attention_multiplier", "output_state", "BLOCK_SIZE_S", "BLOCK_SIZE_V"))
def _linear_attention_backward(
    attention_multiplier: float,
    output_state: bool,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
    residuals: tuple,
    cotangents: tuple,
) -> tuple:
    q, k, v, log_f_cumsum, h0 = residuals
    dy, dht = cotangents
    dht = dht if output_state else None

    B, S, Nq, K = q.shape
    Nk = k.shape[2]
    Nv, V = v.shape[-2:]
    Nf = 0 if log_f_cumsum is None else log_f_cumsum.shape[2]

    N = max(Nq, Nk, Nv, Nf)

    Gq = N // Nq
    Gk = N // Nk
    Gv = N // Nv

    h = _linear_attention_state_passing_core(
        k=k, v=v, log_f_cumsum=log_f_cumsum, h0=h0, N=N, BLOCK_SIZE_S=BLOCK_SIZE_S, BLOCK_SIZE_V=BLOCK_SIZE_V
    )

    dq, dk, dv, dlog_f, dh0 = _linear_attention_backward_core(
        q=q,
        k=k,
        v=v,
        log_f_cumsum=log_f_cumsum,
        h=h,
        dy=dy,
        dht=dht,
        attention_multiplier=attention_multiplier,
        BLOCK_SIZE_S=BLOCK_SIZE_S,
        BLOCK_SIZE_V=BLOCK_SIZE_V,
    )

    dq = dq.reshape(B, S, Nq, Gq, K).sum(axis=3)
    dk = dk.reshape(B, S, Nk, Gk, K).sum(axis=3)
    dv = dv.reshape(B, S, Nv, Gv, V).sum(axis=3)

    if log_f_cumsum is not None:
        Gf = N // Nf
        dlog_f = dlog_f.reshape(B, S, Nf, Gf, *dlog_f.shape[3:]).sum(axis=3)

    if h0 is None:
        dh0 = None

    return dq, dk, dv, dlog_f, dh0


_linear_attention_pallas.defvjp(_linear_attention_forward, _linear_attention_backward)


def _linear_attention_pallas_chunked(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_f: jax.Array | None,
    h0: jax.Array | None,
    attention_multiplier: float,
    output_state: bool,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
) -> tuple[jax.Array, jax.Array]:
    Nq = q.shape[-2]
    Nk = k.shape[-2]
    Nv = v.shape[-2]
    Nf = 0 if log_f is None else log_f.shape[2]

    N = max(Nq, Nk, Nv, Nf)

    Gq = N // Nq
    Gk = N // Nk
    Gv = N // Nv
    Gf = None if log_f is None else N // Nf

    groups = [(Gq, "query"), (Gk, "key"), (Gv, "value")]
    if log_f is not None:
        groups.append((Gf, "log_forget"))

    for G, name in groups:
        if _MAX_HEADS_PER_PALLAS_CELL % G != 0:
            raise ValueError(
                f"grouped head layout with a {name} group size of {G} cannot be split across "
                f"{_MAX_HEADS_PER_PALLAS_CELL}-head chunks (N={N}, Nq={Nq}, Nk={Nk}, Nv={Nv}, Nf={Nf}); "
                "choose q/k/v head counts whose group sizes all divide "
                f"{_MAX_HEADS_PER_PALLAS_CELL}, or use KernelBackend.jax"
            )

    NUM_CHUNKS = ceil_divide(N, _MAX_HEADS_PER_PALLAS_CELL)

    y = []
    ht = []

    for i in range(NUM_CHUNKS):
        start = i * _MAX_HEADS_PER_PALLAS_CELL
        end = min(N, start + _MAX_HEADS_PER_PALLAS_CELL)

        _log_f = None
        if log_f is not None:
            head_slice = slice(start // Gf, end // Gf)
            _log_f = log_f[..., head_slice] if log_f.ndim == 3 else log_f[..., head_slice, :]

        _y, _ht = _linear_attention_pallas(
            q=q[..., start // Gq : end // Gq, :],
            k=k[..., start // Gk : end // Gk, :],
            v=v[..., start // Gv : end // Gv, :],
            log_f=_log_f,
            h0=None if h0 is None else h0[:, start:end],
            attention_multiplier=attention_multiplier,
            output_state=output_state,
            BLOCK_SIZE_S=BLOCK_SIZE_S,
            BLOCK_SIZE_V=BLOCK_SIZE_V,
        )

        y.append(_y)
        ht.append(_ht)

    y = jnp.concatenate(y, axis=2)
    ht = jnp.concatenate(ht, axis=1) if output_state else None

    return y, ht
