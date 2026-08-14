# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

"""Custom-VJP entry point for the pallas linear-attention kernels.

All kernels consume and produce the native (B, S, N, K) host layout —
by design this module performs NO host-level transposes: materializing
(B, N, S, K) copies of q/k/v/dy on every call would cost ~0.74 ms per
forward / ~1.6 ms per fwd+bwd step at B8/S4096/N16/K128/V128 on v6e-1,
and would delay the DMA of the custom backward's first tile by ~0.5 ms.
"""

from functools import partial

import jax

from .backward import _linear_attention_backward_core
from .forward import _linear_attention_forward_core
from .state_passing import _state_passing_core


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def _linear_attention_pallas(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    h0: jax.Array | None,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
) -> tuple[jax.Array, jax.Array]:
    return _linear_attention_forward_core(
        q=q,
        k=k,
        v=v,
        h0=h0,
        attention_multiplier=attention_multiplier,
        BLOCK_SIZE_S=BLOCK_SIZE_S,
        BLOCK_SIZE_V=BLOCK_SIZE_V,
    )


def _linear_attention_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    h0: jax.Array | None,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
) -> tuple[tuple[jax.Array, jax.Array], tuple]:
    y, h = _linear_attention_pallas(
        q=q,
        k=k,
        v=v,
        h0=h0,
        attention_multiplier=attention_multiplier,
        BLOCK_SIZE_S=BLOCK_SIZE_S,
        BLOCK_SIZE_V=BLOCK_SIZE_V,
    )

    return (y, h), (q, k, v, h0)


@partial(jax.jit, static_argnames=("attention_multiplier", "BLOCK_SIZE_S", "BLOCK_SIZE_V"))
def _linear_attention_backward(
    attention_multiplier: float, BLOCK_SIZE_S: int, BLOCK_SIZE_V: int, residuals: tuple, cotangents: tuple
) -> tuple:
    q, k, v, h0 = residuals
    dy, dht = cotangents

    B, S, Nq, K = q.shape
    Nk = k.shape[2]
    Nv, V = v.shape[2], v.shape[-1]

    N = max(Nq, Nk, Nv)

    Gq = N // Nq
    Gk = N // Nk
    Gv = N // Nv

    h = _state_passing_core(k=k, v=v, h0=h0, N=N, BLOCK_SIZE_S=BLOCK_SIZE_S, BLOCK_SIZE_V=BLOCK_SIZE_V)

    dq, dk, dv, dh0 = _linear_attention_backward_core(
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

    dq = dq.reshape(B, S, Nq, Gq, K).sum(axis=3)
    dk = dk.reshape(B, S, Nk, Gk, K).sum(axis=3)
    dv = dv.reshape(B, S, Nv, Gv, V).sum(axis=3)

    if h0 is None:
        dh0 = None

    return dq, dk, dv, dh0


_linear_attention_pallas.defvjp(_linear_attention_forward, _linear_attention_backward)
