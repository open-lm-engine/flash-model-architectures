# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from .....layers_jax.linear_attention.pallas_implementation.backward import _backward_core as _backward_core_jax


def _backward_output_shape_dtype_fn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    dy: torch.Tensor,
    dh: torch.Tensor | None,
) -> list[tuple[tuple[int, ...], torch.dtype]]:
    B, _, S, K = q.shape
    V = v.shape[-1]
    N = dy.shape[1]

    return [
        ((B, N, S, K), q.dtype),
        ((B, N, S, K), q.dtype),
        ((B, N, S, V), q.dtype),
        ((B, N, K, V), torch.float32),
    ]


_BACKWARD_CACHE = {}


def _backward_core(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    dy: torch.Tensor,
    dh: torch.Tensor | None,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # q, k, v, h, dy: already transposed to (B, N, S, K/V). dh: (B, N, K, V) or None - None skips the HBM
    # read/zero-fill entirely and seeds the running state-gradient on-chip instead (see the jax-side
    # kernel, _backward_kernel_zero_dh)
    cache_key = dh is None

    if cache_key not in _BACKWARD_CACHE:
        from torch_xla.experimental.custom_kernel import make_kernel_from_pallas

        _BACKWARD_CACHE[cache_key] = make_kernel_from_pallas(_backward_core_jax, _backward_output_shape_dtype_fn)

    return _BACKWARD_CACHE[cache_key](
        q,
        k,
        v,
        h,
        dy,
        dh,
        static_argnames=("attention_multiplier", "BLOCK_SIZE_S", "BLOCK_SIZE_V"),
        attention_multiplier=attention_multiplier,
        BLOCK_SIZE_S=BLOCK_SIZE_S,
        BLOCK_SIZE_V=BLOCK_SIZE_V,
    )
