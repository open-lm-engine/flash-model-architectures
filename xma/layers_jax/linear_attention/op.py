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
    """exact head-space split for head counts above `_MAX_HEADS_PER_PALLAS_CELL`.

    Linear-attention heads do not interact (the state h is per head), so slicing q/k/v/h0 along
    their grouped head axes, running the kernel on each chunk and concatenating reproduces the
    un-split computation bit for bit. The split is exact iff every chunk boundary lands on a
    group boundary: a head spanning a chunk boundary would have to be included (duplicated) in
    every chunk it overlaps, and the kernel's chunk-local group mapping `n // (chunk_N // chunk_Nx)`
    would then diverge from the global `(lo + n) // Gx` — the guard below rejects every group
    size that does not divide the chunk size, so this never happens.
    """

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
        lo = i * _MAX_HEADS_PER_PALLAS_CELL
        hi = min(N, lo + _MAX_HEADS_PER_PALLAS_CELL)

        y_chunk, ht_chunk = _linear_attention_pallas(
            q=q[:, :, lo // Gq : hi // Gq],
            k=k[:, :, lo // Gk : hi // Gk],
            v=v[:, :, lo // Gv : hi // Gv],
            h0=None if h0 is None else h0[:, lo:hi],
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
    *,
    BLOCK_SIZE_S: int = 256,
    BLOCK_SIZE_V: int = 128,
    kernel_backend: KernelBackend | None = None,
) -> tuple[jax.Array, jax.Array]:
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
    :return: output tensor of shape (B, S, N, V) and output state of shape (B, N, K, V)
    :rtype: tuple[jax.Array, jax.Array]
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
        # support envelope (see the parameter docs above); raising host-side keeps every silent-
        # miscompile class out of the kernels, which run with bounds checks disabled
        if S == 0 or K == 0 or V == 0:
            raise ValueError(
                "linear_attention_jax with KernelBackend.pallas requires S >= 1, K >= 1, and V >= 1; "
                "degenerate zero-length dimensions are not supported by the TPU kernels"
            )
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

        # ragged S: zero-pad to a multiple of BLOCK_SIZE_S at the host. Padded rows are causally
        # isolated (they sit after all real rows), contribute zeros to the final state, and are
        # sliced off below; gradients through jnp.pad/slicing reduce to slicing automatically.
        S_pad = ceil_divide(S, BLOCK_SIZE_S) * BLOCK_SIZE_S - S
        if S_pad != 0:
            pad = ((0, 0), (0, S_pad), (0, 0), (0, 0))
            query = jnp.pad(query, pad)
            key = jnp.pad(key, pad)
            value = jnp.pad(value, pad)

        # feature dims shorter than the 128 TPU lanes, or not a multiple of the lane count /
        # BLOCK_SIZE_V: zero-pad at the host. Every contraction in the kernels is column-local in
        # K/V, so padded columns hold exact zeros, cannot contaminate real entries (unlike a
        # zero-multiplied OOB read), and are sliced off below. This keeps the kernels mask-free —
        # Mosaic does not implement the corresponding small-shape broadcasts.
        K_width = ceil_divide(K, _TPU_LANE_COUNT) * _TPU_LANE_COUNT
        K_pad = K_width - K
        # V is padded to a multiple of BLOCK_SIZE_V unconditionally: the kernels require
        # V % BLOCK_SIZE_V == 0, and BLOCK_SIZE_V can exceed 128 while V sits between the
        # 128-lane round-up and BLOCK_SIZE_V (e.g. V = 100 with BLOCK_SIZE_V = 256).
        V_width = ceil_divide(V, BLOCK_SIZE_V) * BLOCK_SIZE_V  # >= BLOCK_SIZE_V, a 128 multiple
        V_pad = V_width - V
        if K_pad != 0:
            pad_K = ((0, 0), (0, 0), (0, 0), (0, K_pad))
            query = jnp.pad(query, pad_K)
            key = jnp.pad(key, pad_K)
        if V_pad != 0:
            value = jnp.pad(value, ((0, 0), (0, 0), (0, 0), (0, V_pad)))
        if (K_pad != 0 or V_pad != 0) and input_state is not None:
            input_state = jnp.pad(input_state, ((0, 0), (0, 0), (0, K_pad), (0, V_pad)))

        if N <= _MAX_HEADS_PER_PALLAS_CELL:
            y, ht = _linear_attention_pallas(
                q=query,
                k=key,
                v=value,
                h0=input_state,
                attention_multiplier=attention_multiplier,
                BLOCK_SIZE_S=BLOCK_SIZE_S,
                BLOCK_SIZE_V=BLOCK_SIZE_V,
            )
        else:
            y, ht = _linear_attention_pallas_chunked(
                q=query,
                k=key,
                v=value,
                h0=input_state,
                attention_multiplier=attention_multiplier,
                BLOCK_SIZE_S=BLOCK_SIZE_S,
                BLOCK_SIZE_V=BLOCK_SIZE_V,
            )

        if S_pad != 0:
            y = y[:, :S]
        if V_pad != 0:
            y = y[:, :, :, :V]
        if K_pad != 0 or V_pad != 0:
            ht = ht[:, :, :K, :V]
    elif kernel_backend == KernelBackend.jax:
        y, ht = _linear_attention_reference(
            q=query, k=key, v=value, h0=input_state, attention_multiplier=attention_multiplier
        )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return y, ht
