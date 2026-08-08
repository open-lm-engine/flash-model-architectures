# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ....layers_jax.depthwise_causal_convolution.pallas_implementation.forward import (
    _forward_core as _depthwise_causal_convolution_forward_core_jax,
)


def _metadata(
    x: torch.Tensor, W: torch.Tensor, b: torch.Tensor | None, h0: torch.Tensor | None
) -> list[tuple[tuple[int, ...], torch.dtype]]:
    B, S, H = x.shape

    return [((B, S, H), x.dtype)]


_CACHE = {}


def _forward_core(
    x: torch.Tensor, W: torch.Tensor, b: torch.Tensor | None, h0: torch.Tensor | None, BLOCK_SIZE_S: int
) -> torch.Tensor:
    cache_key = (b is None, h0 is None)

    if cache_key not in _CACHE:
        from torch_xla.experimental.custom_kernel import make_kernel_from_pallas

        _CACHE[cache_key] = make_kernel_from_pallas(_depthwise_causal_convolution_forward_core_jax, _metadata)

    return _CACHE[cache_key](x, W, b, h0, static_argnames=("BLOCK_SIZE_S",), BLOCK_SIZE_S=BLOCK_SIZE_S)
