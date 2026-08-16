# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from functools import partial

import jax
import jax.numpy as jnp

from ....math import ceil_divide
from .backward import _linear_attention_backward_core
from .forward import _linear_attention_forward_core
from .state_passing import _linear_attention_state_passing_core


_MAX_HEADS_PER_PALLAS_CELL = 16


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7))
def _linear_attention_pallas(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    h0: jax.Array | None,
    attention_multiplier: float,
    output_state: bool,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
) -> tuple[jax.Array, jax.Array]:
    return _linear_attention_forward_core(
        q=q,
        k=k,
        v=v,
        h0=h0,
        attention_multiplier=attention_multiplier,
        output_state=output_state,
        BLOCK_SIZE_S=BLOCK_SIZE_S,
        BLOCK_SIZE_V=BLOCK_SIZE_V,
    )


def _linear_attention_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    h0: jax.Array | None,
    attention_multiplier: float,
    output_state: bool,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
) -> tuple[tuple[jax.Array, jax.Array | None], tuple]:
    y, h = _linear_attention_pallas(
        q=q,
        k=k,
        v=v,
        h0=h0,
        attention_multiplier=attention_multiplier,
        output_state=output_state,
        BLOCK_SIZE_S=BLOCK_SIZE_S,
        BLOCK_SIZE_V=BLOCK_SIZE_V,
    )

    return (y, h), (q, k, v, h0)


@partial(jax.jit, static_argnames=("attention_multiplier", "output_state", "BLOCK_SIZE_S", "BLOCK_SIZE_V"))
def _linear_attention_backward(
    attention_multiplier: float,
    output_state: bool,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
    residuals: tuple,
    cotangents: tuple,
) -> tuple:
    q, k, v, h0 = residuals
    dy, dht = cotangents
    dht = dht if output_state else None

    B, S, Nq, K = q.shape
    Nk = k.shape[2]
    Nv, V = v.shape[-2:]

    N = max(Nq, Nk, Nv)

    Gq = N // Nq
    Gk = N // Nk
    Gv = N // Nv

    h = _linear_attention_state_passing_core(
        k=k, v=v, h0=h0, N=N, output_state=False, BLOCK_SIZE_S=BLOCK_SIZE_S, BLOCK_SIZE_V=BLOCK_SIZE_V
    )

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


def _linear_attention_pallas_chunked(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    h0: jax.Array | None,
    attention_multiplier: float,
    output_state: bool,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
) -> tuple[jax.Array, jax.Array]:
    Nq = q.shape[-2]
    Nk = k.shape[-2]
    Nv = v.shape[-2]
    N = max(Nq, Nk, Nv)

    Gq = N // Nq
    Gk = N // Nk
    Gv = N // Nv

    for G, name in ((Gq, "query"), (Gk, "key"), (Gv, "value")):
        if _MAX_HEADS_PER_PALLAS_CELL % G != 0:
            raise ValueError(
                f"grouped head layout with a {name} group size of {G} cannot be split across "
                f"{_MAX_HEADS_PER_PALLAS_CELL}-head chunks (N={N}, Nq={Nq}, Nk={Nk}, Nv={Nv}); "
                "choose q/k/v head counts whose group sizes all divide "
                f"{_MAX_HEADS_PER_PALLAS_CELL}, or use KernelBackend.jax"
            )

    NUM_CHUNKS = ceil_divide(N, _MAX_HEADS_PER_PALLAS_CELL)

    y = []
    ht = []

    for i in range(NUM_CHUNKS):
        start = i * _MAX_HEADS_PER_PALLAS_CELL
        end = min(N, start + _MAX_HEADS_PER_PALLAS_CELL)

        _y, _ht = _linear_attention_pallas(
            q=q[..., start // Gq : end // Gq, :],
            k=k[..., start // Gk : end // Gk, :],
            v=v[..., start // Gv : end // Gv, :],
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
