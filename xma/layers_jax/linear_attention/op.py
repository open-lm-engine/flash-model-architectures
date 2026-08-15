# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import math

import jax

from ...accelerator import Accelerator, KernelBackend
from .jax_implementation import _linear_attention_reference
from .pallas_implementation import _linear_attention_pallas


def _get_num_heads(q: jax.Array, k: jax.Array, v: jax.Array) -> tuple[int, int, int, int]:
    Nq = q.shape[-2]
    Nk = k.shape[-2]
    Nv = v.shape[-2]

    N = max(Nq, Nk, Nv)

    assert N % Nq == 0
    assert N % Nk == 0
    assert N % Nv == 0

    return Nq, Nk, Nv, N


def linear_attention_jax(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    input_state: jax.Array | None = None,
    attention_multiplier: float | None = None,
    output_state: bool = True,
    *,
    BLOCK_SIZE_S: int = 256,
    BLOCK_SIZE_V: int = 256,
    kernel_backend: KernelBackend | None = None,
) -> tuple[jax.Array, jax.Array | None]:
    """computes linear attention: `y[s] = q[s] @ h[s]`, `h[s] = h[s - 1] + k[s].T @ v[s]`

    :param query: query tensor of shape (B, S, Nq, K)
    :type query: jax.Array
    :param key: key tensor of shape (B, S, Nk, K)
    :type key: jax.Array
    :param value: value tensor of shape (B, S, Nv, V)
    :type value: jax.Array
    :param input_state: starting state of shape (B, N, K, V), where N = max{Nq, Nk, Nv}. None means starting
        state is 0 tensor. Defaults to None.
    :type input_state: jax.Array | None
    :param attention_multiplier: scaling factor applied to the output, `y`. None defaults to `1 / sqrt(K)`.
        Defaults to None.
    :type attention_multiplier: float | None
    :param output_state: whether to also return the final state (of shape (B, N, K, V)) for use as
        `input_state` in a subsequent call. When False, the second return value is None. Defaults to True.
    :type output_state: bool
    :param BLOCK_SIZE_S: sequence-length block size used by the pallas kernel. Defaults to 256.
    :type BLOCK_SIZE_S: int
    :param BLOCK_SIZE_V: value-head-dimension block size used by the pallas kernel; `V <= BLOCK_SIZE_V`
        (the default) means V is effectively untiled. Mosaic requires this to be a multiple of 256 or
        exactly equal to `V` - other values raise at trace time. Defaults to 256.
    :type BLOCK_SIZE_V: int
    :param kernel_backend: KernelBackend
    :type kernel_backend: KernelBackend | None
    :return: output tensor of shape (B, S, N, V), and output state of shape (B, N, K, V) if `output_state`
        is True else None.
    :rtype: tuple[jax.Array, jax.Array | None]
    """

    B, S, _, K = query.shape
    V = value.shape[-1]

    Nq, Nk, Nv, N = _get_num_heads(query, key, value)

    assert query.shape == (B, S, Nq, K)
    assert key.shape == (B, S, Nk, K)
    assert value.shape == (B, S, Nv, V)

    if input_state is not None:
        assert input_state.shape == (B, N, K, V)

    if attention_multiplier is None:
        attention_multiplier = 1 / math.sqrt(K)

    if kernel_backend is None:
        kernel_backend = Accelerator.get_kernel_backend()

    if kernel_backend == KernelBackend.pallas:
        y, ht = _linear_attention_pallas(
            q=query,
            k=key,
            v=value,
            h0=input_state,
            attention_multiplier=attention_multiplier,
            output_state=output_state,
            BLOCK_SIZE_S=BLOCK_SIZE_S,
            BLOCK_SIZE_V=BLOCK_SIZE_V,
        )
    elif kernel_backend == KernelBackend.jax:
        y, ht = _linear_attention_reference(
            q=query,
            k=key,
            v=value,
            h0=input_state,
            attention_multiplier=attention_multiplier,
            output_state=output_state,
        )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return y, ht
