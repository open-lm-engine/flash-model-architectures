# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from functools import partial

import jax

from .backward import _linear_attention_backward_pallas
from .forward import _linear_attention_forward_pallas


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def _linear_attention_jax_op(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    h0: jax.Array | None,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
) -> tuple[jax.Array, jax.Array]:
    return _linear_attention_forward_pallas(
        q=q,
        k=k,
        v=v,
        h0=h0,
        attention_multiplier=attention_multiplier,
        BLOCK_SIZE_S=BLOCK_SIZE_S,
        BLOCK_SIZE_V=BLOCK_SIZE_V,
    )


def _linear_attention_forward_jax(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    input_state: jax.Array | None,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
) -> tuple[tuple[jax.Array, jax.Array], tuple]:
    y, h = _linear_attention_jax_op(
        q=query,
        k=key,
        v=value,
        h0=input_state,
        attention_multiplier=attention_multiplier,
        BLOCK_SIZE_S=BLOCK_SIZE_S,
        BLOCK_SIZE_V=BLOCK_SIZE_V,
    )

    return (y, h), (query, key, value, input_state)


def _linear_attention_backward_jax(
    attention_multiplier: float, BLOCK_SIZE_S: int, BLOCK_SIZE_V: int, residuals: tuple, cotangents: tuple
) -> tuple:
    query, key, value, input_state = residuals
    dy, dht = cotangents

    dq, dk, dv, dh0 = _linear_attention_backward_pallas(
        q=query,
        k=key,
        v=value,
        dy=dy,
        h0=input_state,
        dht=dht,
        attention_multiplier=attention_multiplier,
        BLOCK_SIZE_S=BLOCK_SIZE_S,
        BLOCK_SIZE_V=BLOCK_SIZE_V,
    )

    if input_state is None:
        dh0 = None

    return dq, dk, dv, dh0


_linear_attention_jax_op.defvjp(_linear_attention_forward_jax, _linear_attention_backward_jax)
