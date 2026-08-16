# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import math

import jax
import jax.numpy as jnp

from ...accelerator import Accelerator, KernelBackend
from ...math import ceil_divide
from .jax_implementation import _linear_attention_reference
from .pallas_implementation import (
    _MAX_HEADS_PER_PALLAS_CELL,
    _linear_attention_pallas,
    _linear_attention_pallas_chunked,
)


def linear_attention_jax(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    forget: jax.Array | None = None,
    input_state: jax.Array | None = None,
    attention_multiplier: float | None = None,
    output_state: bool = True,
    *,
    BLOCK_SIZE_S: int = 256,
    BLOCK_SIZE_V: int = 128,
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
    :param BLOCK_SIZE_S: sequence-length block size used by the pallas kernel. Must be >= 256 for
        KernelBackend.pallas (support envelope, not a numerical requirement: the kernels are shipped
        and validated at this envelope only). Defaults to 256.
    :type BLOCK_SIZE_S: int
    :param BLOCK_SIZE_V: value-head-dimension block size used by the pallas kernel; `V <= BLOCK_SIZE_V`
        (the default) means V is effectively untiled. Must be a multiple of 128 (the TPU lane
        count); `V` is zero-padded to a multiple of `BLOCK_SIZE_V` host-side when needed.
        The VMEM state scratches are (heads, K, ceil(V / BLOCK_SIZE_V) * BLOCK_SIZE_V) float32,
        i.e. heads x K x padded-V x 4 bytes ~ 16 * 128 * V * 4 B at the production envelope
        (about 8 MB at V = 1024), so do not point this knob at extreme V widths; it is intended
        for ordinary head dims up to a few hundred. Defaults to 128.
    :type BLOCK_SIZE_V: int
    :param kernel_backend: KernelBackend
    :type kernel_backend: KernelBackend | None
    :return: output tensor of shape (B, S, N, V), and output state of shape (B, N, K, V) if `output_state`
        is True else None.
    :rtype: tuple[jax.Array, jax.Array | None]
    """

    B, S, _, K = query.shape
    V = value.shape[-1]

    Nq = query.shape[-2]
    Nk = key.shape[-2]
    Nv = value.shape[-2]

    N = max(Nq, Nk, Nv)

    assert N % Nq == 0
    assert N % Nk == 0
    assert N % Nv == 0

    assert query.shape == (B, S, Nq, K)
    assert key.shape == (B, S, Nk, K)
    assert value.shape == (B, S, Nv, V)

    if input_state is not None:
        assert input_state.shape == (B, N, K, V)

    if attention_multiplier is None:
        attention_multiplier = 1 / math.sqrt(K)

    if kernel_backend is None:
        kernel_backend = Accelerator.get_kernel_backend()

    if S == 1 or kernel_backend == KernelBackend.jax:
        y, ht = _linear_attention_reference(
            q=query,
            k=key,
            v=value,
            f=forget,
            h0=input_state,
            attention_multiplier=attention_multiplier,
            output_state=output_state,
        )
    elif kernel_backend == KernelBackend.pallas:
        lane_count = Accelerator.get_lane_count()
        BLOCK_SIZE_K = lane_count

        if BLOCK_SIZE_V <= 0 or BLOCK_SIZE_V % lane_count != 0:
            raise ValueError(
                f"BLOCK_SIZE_V ({BLOCK_SIZE_V}) must be a positive multiple of {lane_count} "
                "(the TPU lane count) for KernelBackend.pallas"
            )

        S_pad = ceil_divide(S, BLOCK_SIZE_S) * BLOCK_SIZE_S - S
        K_pad = ceil_divide(K, BLOCK_SIZE_K) * BLOCK_SIZE_K - K
        V_pad = ceil_divide(V, BLOCK_SIZE_V) * BLOCK_SIZE_V - V

        if S_pad != 0 or K_pad != 0:
            pad = ((0, 0), (0, S_pad), (0, 0), (0, K_pad))
            query = jnp.pad(query, pad)
            key = jnp.pad(key, pad)

        if S_pad != 0 or V_pad != 0:
            value = jnp.pad(value, ((0, 0), (0, S_pad), (0, 0), (0, V_pad)))

        if input_state is not None and (K_pad != 0 or V_pad != 0):
            input_state = jnp.pad(input_state, ((0, 0), (0, 0), (0, K_pad), (0, V_pad)))

        y, ht = (_linear_attention_pallas if N <= _MAX_HEADS_PER_PALLAS_CELL else _linear_attention_pallas_chunked)(
            q=query,
            k=key,
            v=value,
            f=forget,
            h0=input_state,
            attention_multiplier=attention_multiplier,
            output_state=output_state,
            BLOCK_SIZE_S=BLOCK_SIZE_S,
            BLOCK_SIZE_V=BLOCK_SIZE_V,
        )

        y = y[:, :S, :, :V]

        if ht is not None:
            ht = ht[..., :K, :V]
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return y, ht
