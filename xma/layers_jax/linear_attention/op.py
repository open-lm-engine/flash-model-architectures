# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import math

import jax
import jax.numpy as jnp

from ...accelerator import Accelerator, KernelBackend
from ...math import ceil_divide
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


# the kernels visit heads via a static python loop unrolled at trace time, and the dh / state
# scratches are (heads, K, NUM_V_TILES * BLOCK_SIZE_V) f32; 16 heads keeps the unrolled graph,
# register pressure, and scratch VMEM bounded at the validated production tile sizes.
_MAX_HEADS_PER_PALLAS_CELL = 16
_TPU_LANE_COUNT = 128  # Pallas kernel trailing dims are padded to the TPU lane count
_MIN_BLOCK_SIZE_S = 256  # lowest S tile the shipped kernels are validated at


def _linear_attention_pallas_chunked(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    h0: jax.Array | None,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
) -> tuple[jax.Array, jax.Array]:
    Nq, Nk, Nv, N = _get_num_heads(q, k, v)
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

    ys = []
    hts = []
    for i in range(NUM_CHUNKS):
        start = i * _MAX_HEADS_PER_PALLAS_CELL
        end = min(N, start + _MAX_HEADS_PER_PALLAS_CELL)

        y_chunk, ht_chunk = _linear_attention_pallas(
            q=q[:, :, start // Gq : end // Gq],
            k=k[:, :, start // Gk : end // Gk],
            v=v[:, :, start // Gv : end // Gv],
            h0=None if h0 is None else h0[:, start:end],
            attention_multiplier=attention_multiplier,
            BLOCK_SIZE_S=BLOCK_SIZE_S,
            BLOCK_SIZE_V=BLOCK_SIZE_V,
        )
        ys.append(y_chunk)
        hts.append(ht_chunk)

    return jnp.concatenate(ys, axis=2), jnp.concatenate(hts, axis=1)


def linear_attention_jax(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
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
        if BLOCK_SIZE_S < _MIN_BLOCK_SIZE_S:
            raise ValueError(
                f"BLOCK_SIZE_S ({BLOCK_SIZE_S}) must be >= {_MIN_BLOCK_SIZE_S} for KernelBackend.pallas "
                "(validated support envelope of the shipped kernels; there is no in-kernel handling below it)"
            )

        if BLOCK_SIZE_V <= 0 or BLOCK_SIZE_V % _TPU_LANE_COUNT != 0:
            raise ValueError(
                f"BLOCK_SIZE_V ({BLOCK_SIZE_V}) must be a positive multiple of {_TPU_LANE_COUNT} "
                "(the TPU lane count) for KernelBackend.pallas"
            )

        S_pad = ceil_divide(S, BLOCK_SIZE_S) * BLOCK_SIZE_S - S
        if S_pad != 0:
            pad = ((0, 0), (0, S_pad), (0, 0), (0, 0))
            query = jnp.pad(query, pad)
            key = jnp.pad(key, pad)
            value = jnp.pad(value, pad)

        K_pad = ceil_divide(K, _TPU_LANE_COUNT) * _TPU_LANE_COUNT - K
        V_pad = ceil_divide(V, BLOCK_SIZE_V) * BLOCK_SIZE_V - V

        if K_pad != 0:
            pad_K = ((0, 0), (0, 0), (0, 0), (0, K_pad))
            query = jnp.pad(query, pad_K)
            key = jnp.pad(key, pad_K)

        if V_pad != 0:
            value = jnp.pad(value, ((0, 0), (0, 0), (0, 0), (0, V_pad)))

        if input_state is not None and (K_pad != 0 or V_pad != 0):
            input_state = jnp.pad(input_state, ((0, 0), (0, 0), (0, K_pad), (0, V_pad)))

        y, ht = (_linear_attention_pallas if N <= _MAX_HEADS_PER_PALLAS_CELL else _linear_attention_pallas_chunked)(
            q=query,
            k=key,
            v=value,
            h0=input_state,
            attention_multiplier=attention_multiplier,
            BLOCK_SIZE_S=BLOCK_SIZE_S,
            BLOCK_SIZE_V=BLOCK_SIZE_V,
        )

        y = y[:, :S]
        y = y[:, :, :, :V]
        ht = ht[:, :, :K, :V]
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
