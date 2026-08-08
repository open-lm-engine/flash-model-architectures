# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from .backward import _backward_core, _state_passing_core
from .forward import _forward_core


@partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _depthwise_causal_convolution_pallas(
    x: jax.Array,
    W: jax.Array,
    b: jax.Array | None,
    h0: jax.Array | None,
    output_state: bool,
    BLOCK_SIZE_S: int,
) -> tuple[jax.Array, jax.Array | None]:
    W = jnp.transpose(W, (1, 0))
    b = None if b is None else b.astype(jnp.float32)[None, :]

    if h0 is not None:
        h0 = jnp.transpose(h0, (0, 2, 1)).astype(x.dtype)

    y = _forward_core(x=x, W=W, b=b, h0=h0, BLOCK_SIZE_S=BLOCK_SIZE_S)

    if not output_state:
        return y, None

    state_size = W.shape[0] - 1
    if h0 is None:
        ht = (
            jnp.pad(x, ((0, 0), (state_size - x.shape[1], 0), (0, 0)))
            if x.shape[1] < state_size
            else x[:, -state_size:, :]
        )
    else:
        ht = jnp.concatenate([h0.astype(x.dtype), x], axis=1)[:, -state_size:, :]

    return y, jnp.transpose(ht.astype(jnp.float32), (0, 2, 1))


def _depthwise_causal_convolution_forward(
    x: jax.Array,
    W: jax.Array,
    b: jax.Array | None,
    h0: jax.Array | None,
    output_state: bool,
    BLOCK_SIZE_S: int,
) -> tuple[tuple[jax.Array, jax.Array | None], tuple]:
    y, ht = _depthwise_causal_convolution_pallas(
        x=x, W=W, b=b, h0=h0, output_state=output_state, BLOCK_SIZE_S=BLOCK_SIZE_S
    )

    return (y, ht), (x, W, b, h0)


def _depthwise_causal_convolution_backward(
    output_state: bool, BLOCK_SIZE_S: int, residuals: tuple, cotangents: tuple
) -> tuple:
    x, W, b, h0 = residuals
    dy, dht = cotangents

    B, _, H = x.shape
    K = W.shape[-1]

    W = jnp.transpose(W, (1, 0))

    if h0 is not None:
        h0 = jnp.transpose(h0, (0, 2, 1)).astype(x.dtype)

    dht = None if dht is None or not output_state else jnp.transpose(dht, (0, 2, 1)).astype(jnp.float32)

    h = _state_passing_core(x=x, h0=h0, BLOCK_SIZE_S=BLOCK_SIZE_S, K=K)
    dx, dW, db, dh0 = _backward_core(x=x, W=W, h=h, dy=dy, dht=dht, BLOCK_SIZE_S=BLOCK_SIZE_S, K=K)

    dW = jnp.transpose(dW, (1, 0))
    db = None if b is None else db[0]
    dh0 = None if h0 is None else jnp.transpose(dh0, (0, 2, 1))

    return dx, dW, db, dh0


_depthwise_causal_convolution_pallas.defvjp(
    _depthwise_causal_convolution_forward, _depthwise_causal_convolution_backward
)


_BASE_ACTIVATIONS = {
    "gelu": jax.nn.gelu,
    "relu": jax.nn.relu,
    "sigmoid": jax.nn.sigmoid,
    "silu": jax.nn.silu,
    "swish": jax.nn.silu,
    "tanh": jnp.tanh,
}


def _get_activation_function(name: str | None) -> Callable[[jax.Array], jax.Array]:
    if name is None:
        return lambda x: x

    if name not in _BASE_ACTIVATIONS:
        raise ValueError(f"invalid activation function ({name})")

    return _BASE_ACTIVATIONS[name]


def _apply_mask_to_padding_states(x: jax.Array, attention_mask: jax.Array | None) -> jax.Array:
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        x = (x * attention_mask[:, :, None]).astype(x.dtype)

    return x
