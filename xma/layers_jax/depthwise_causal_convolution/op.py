# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from typing import Callable

import jax

from ...accelerator import Accelerator, KernelBackend
from .jax_implementation import _depthwise_causal_convolution_reference
from .pallas_implementation import (
    _apply_mask_to_padding_states,
    _depthwise_causal_convolution_pallas,
    _get_activation_function,
)


def depthwise_causal_convolution_jax(
    input: jax.Array,
    weight: jax.Array,
    bias: jax.Array | None = None,
    input_state: jax.Array | None = None,
    attention_mask: jax.Array | None = None,
    output_state: bool = False,
    activation_function: str | Callable[[jax.Array], jax.Array] | None = None,
    *,
    BLOCK_SIZE_S: int = 128,
    kernel_backend: KernelBackend | None = None,
) -> tuple[jax.Array, jax.Array | None]:
    """
    computes depthwise causal 1D convolution: `output[b, t, h] = act(bias[h] + sum_k weight[h, k] *
    z[b, t + k, h])` where `z` is `input` preceded by `kernel_size - 1` raw history positions taken from
    `input_state` (or 0 if `input_state` is None), i.e. `output[b, t]` only depends on
    `input[b, t - kernel_size + 1 : t + 1]`.

    :param input: input tensor of shape (B, S, H)
    :type input: jax.Array
    :param weight: depthwise convolution weight of shape (H, K), K being the kernel size
    :type weight: jax.Array
    :param bias: bias tensor of shape (H,). None means no bias is added. Defaults to None.
    :type bias: jax.Array | None
    :param input_state: the `K - 1` raw (pre-convolution) input positions preceding `input`, of shape
        (B, H, K - 1). None is equivalent to a 0 tensor. Defaults to None.
    :type input_state: jax.Array | None
    :param attention_mask: mask of shape (B, S), zeroing out padding positions before and after the
        convolution. Defaults to None.
    :type attention_mask: jax.Array | None
    :param output_state: whether to also return the trailing `K - 1` raw input positions (taken from `input`,
        falling back to `input_state` if `input` is shorter than `K - 1`) for use as `input_state` in a
        subsequent call. Defaults to False.
    :type output_state: bool
    :param activation_function: activation applied after the convolution + bias. Either a name in
        {"gelu", "relu", "sigmoid", "silu", "swish", "tanh"}, an arbitrary callable, or None (identity).
        Defaults to None.
    :type activation_function: str | Callable[[jax.Array], jax.Array] | None
    :param kernel_backend: KernelBackend.pallas uses a hand-written VPU-only Pallas TPU kernel (avoids the
        MXU/systolic array, which a tiny `kernel_size` reduction dimension underutilizes badly).
        KernelBackend.jax uses the plain `jax.lax.conv_general_dilated`-based reference. None auto-detects
        based on the accelerator (KernelBackend.pallas on TPU). Defaults to None.
    :type kernel_backend: KernelBackend | None
    :return: output tensor of shape (B, S, H), and the output state of shape (B, H, K - 1) if `output_state`
        is True else None.
    :rtype: tuple[jax.Array, jax.Array | None]
    """

    B, _, H = input.shape
    K = weight.shape[-1]

    assert weight.ndim == 2
    assert weight.shape[0] == H
    assert K > 1

    if bias is not None:
        assert bias.shape == (H,)

    if input_state is not None:
        assert input_state.shape == (B, H, K - 1)

    if activation_function is None or isinstance(activation_function, str):
        activation_function = _get_activation_function(activation_function)

    if kernel_backend is None:
        kernel_backend = Accelerator.get_kernel_backend()

    input = _apply_mask_to_padding_states(input, attention_mask)

    if kernel_backend == KernelBackend.pallas:
        output, final_state = _depthwise_causal_convolution_pallas(
            x=input, W=weight, b=bias, h0=input_state, output_state=output_state, BLOCK_SIZE_S=BLOCK_SIZE_S
        )
    elif kernel_backend == KernelBackend.jax:
        output, final_state = _depthwise_causal_convolution_reference(
            x=input, W=weight, b=bias, h0=input_state, output_state=output_state
        )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    output = activation_function(output)
    output = _apply_mask_to_padding_states(output, attention_mask)

    return output, final_state
