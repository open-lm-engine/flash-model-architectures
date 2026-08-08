# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ....layers_jax.depthwise_causal_convolution.pallas_implementation.forward import (
    _forward_core as _depthwise_causal_convolution_forward_core_jax,
)


def _output_shape_dtype_fn(
    x: torch.Tensor, W: torch.Tensor, b: torch.Tensor | None, h0: torch.Tensor | None, BLOCK_SIZE_S: int
) -> list[tuple[tuple[int, ...], torch.dtype]]:
    B, S, H = x.shape

    return [((B, S, H), x.dtype)]


_CACHE = {}


def _depthwise_causal_convolution_forward_core(
    x: torch.Tensor, W: torch.Tensor, b: torch.Tensor | None, h0: torch.Tensor | None, BLOCK_SIZE_S: int
) -> torch.Tensor:
    cache_key = (b is None, h0 is None)

    if cache_key not in _CACHE:
        from torch_xla.experimental.custom_kernel import make_kernel_from_pallas

        _CACHE[cache_key] = make_kernel_from_pallas(
            _depthwise_causal_convolution_forward_core_jax, _output_shape_dtype_fn
        )

    return _CACHE[cache_key](x, W, b, h0, static_argnames=("BLOCK_SIZE_S",), BLOCK_SIZE_S=BLOCK_SIZE_S)


def _depthwise_causal_convolution_forward_pallas(
    x: torch.Tensor, W: torch.Tensor, b: torch.Tensor | None, h0: torch.Tensor | None, BLOCK_SIZE_S: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    # W: (H, K) -> (K, H); b: (H,) -> (1, H) or None; h0: (B, H, K - 1) -> (B, K - 1, H) or None
    W = W.transpose(1, 0)
    b = None if b is None else b.float()[None, :]
    h0 = None if h0 is None else h0.transpose(1, 2).to(x.dtype)

    y = _depthwise_causal_convolution_forward_core(x, W, b, h0, BLOCK_SIZE_S=BLOCK_SIZE_S)

    state_size = W.shape[0] - 1
    if h0 is None:
        ht = (
            torch.nn.functional.pad(x, (0, 0, state_size - x.shape[1], 0))
            if x.shape[1] < state_size
            else x[:, -state_size:, :]
        )
    else:
        ht = torch.cat((h0, x), dim=1)[:, -state_size:, :]

    return y, ht.float()
