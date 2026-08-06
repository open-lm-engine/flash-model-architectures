# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from functools import partial

import jax
import jax.numpy as jnp

from .backward import _backward_core, _state_passing_core
from .forward import _forward_core


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
    q = jnp.swapaxes(q, 1, 2)
    k = jnp.swapaxes(k, 1, 2)
    v = jnp.swapaxes(v, 1, 2)

    y, ht = _forward_core(
        q=q,
        k=k,
        v=v,
        h0=h0,
        attention_multiplier=attention_multiplier,
        BLOCK_SIZE_S=BLOCK_SIZE_S,
        BLOCK_SIZE_V=BLOCK_SIZE_V,
    )

    y = jnp.swapaxes(y, 1, 2)

    return y, ht


def _linear_attention_forward(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    input_state: jax.Array | None,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
) -> tuple[tuple[jax.Array, jax.Array], tuple]:
    y, h = _linear_attention_pallas(
        q=query,
        k=key,
        v=value,
        h0=input_state,
        attention_multiplier=attention_multiplier,
        BLOCK_SIZE_S=BLOCK_SIZE_S,
        BLOCK_SIZE_V=BLOCK_SIZE_V,
    )

    return (y, h), (query, key, value, input_state)


@partial(jax.jit, static_argnames=("attention_multiplier", "BLOCK_SIZE_S", "BLOCK_SIZE_V"))
def _linear_attention_backward(
    attention_multiplier: float, BLOCK_SIZE_S: int, BLOCK_SIZE_V: int, residuals: tuple, cotangents: tuple
) -> tuple:
    q, k, v, h0 = residuals
    dy, dht = cotangents

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

    if h0 is None:
        dh0 = None

    return dq, dk, dv, dh0


_linear_attention_pallas.defvjp(_linear_attention_forward, _linear_attention_backward)
