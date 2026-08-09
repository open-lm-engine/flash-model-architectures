# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ....layers_jax.linear_attention.pallas_implementation import _linear_attention_backward_core as _backward_core_jax


def _output_shape_dtype_fn(
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


def _linear_attention_backward_pallas(
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
    if not hasattr(_linear_attention_backward_pallas, "cache"):
        _linear_attention_backward_pallas.cache = {}

    cache_key = dh is None
    kernel = _linear_attention_backward_pallas.cache.get(cache_key)

    if kernel is None:
        from torch_xla.experimental.custom_kernel import make_kernel_from_pallas

        kernel = make_kernel_from_pallas(_backward_core_jax, _output_shape_dtype_fn)
        _linear_attention_backward_pallas.cache[cache_key] = kernel

    return kernel(
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
