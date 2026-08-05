# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax

from ...accelerator import Accelerator, KernelBackend


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


def _last_k_columns(xt: jax.Array, K: int) -> jax.Array:
    """Return the last K columns of `xt` along its trailing axis, left-zero-padded if `xt` is shorter than K."""

    S = xt.shape[-1]
    if S < K:
        return jnp.pad(xt, ((0, 0), (0, 0), (K - S, 0)))

    return xt[:, :, -K:]


def _compute_final_state(x: jax.Array, input_state: jax.Array | None, K: int, output_state: bool) -> jax.Array | None:
    # the trailing K raw input positions to hand back as `input_state` to a subsequent call - pure indexing,
    # independent of however `y` itself gets computed, so this is shared unchanged by every kernel_backend.
    if not output_state:
        return None

    xt = jnp.transpose(x, (0, 2, 1))  # (B, H, S)

    if input_state is None:
        return _last_k_columns(xt, K)

    full = jnp.concatenate([input_state.astype(x.dtype), xt], axis=-1)  # (B, H, K + S)
    return full[:, :, -K:]


def _depthwise_causal_convolution_reference(
    x: jax.Array, weight: jax.Array, bias: jax.Array | None, input_state: jax.Array | None, output_state: bool
) -> tuple[jax.Array, jax.Array | None]:
    # decode / multi-token continuation prepend the given K-wide state and take a valid (unpadded)
    # cross-correlation over the concatenation - output position j only depends on
    # full[..., j + 1 : j + 1 + K], i.e. the K raw inputs ending at (and including) position j. Prefill
    # (no state) is mathematically the same formula with an all-zero state, but we ask XLA for the
    # equivalent (K - 1, 0) causal padding directly instead of materializing and concatenating zeros.
    H = x.shape[-1]
    K = weight.shape[-1]

    xt = jnp.transpose(x, (0, 2, 1))  # (B, H, S)

    if input_state is None:
        conv_lhs = xt
        padding = [(K - 1, 0)]
    else:
        full = jnp.concatenate([input_state.astype(x.dtype), xt], axis=-1)  # (B, H, K + S)
        conv_lhs = full[:, :, 1:]
        padding = [(0, 0)]

    y = lax.conv_general_dilated(
        lhs=conv_lhs,
        rhs=weight[:, None, :],
        window_strides=(1,),
        padding=padding,
        feature_group_count=H,
        dimension_numbers=("NCH", "OIH", "NCH"),
    )

    if bias is not None:
        y = y + bias[None, :, None]

    y = jnp.transpose(y, (0, 2, 1))  # (B, S, H)

    final_state = _compute_final_state(x, input_state, K, output_state)

    return y, final_state


@partial(jax.custom_vjp, nondiff_argnums=(4,))
def _depthwise_causal_convolution_pallas_op(
    x: jax.Array, weight: jax.Array, bias: jax.Array | None, input_state: jax.Array | None, BLOCK_SIZE_S: int
) -> jax.Array:
    y, _ = _depthwise_causal_convolution_forward_pallas(x, weight, bias, input_state, BLOCK_SIZE_S)
    return y


def _depthwise_causal_convolution_pallas_fwd(
    x: jax.Array, weight: jax.Array, bias: jax.Array | None, input_state: jax.Array | None, BLOCK_SIZE_S: int
) -> tuple[jax.Array, tuple]:
    y, _ = _depthwise_causal_convolution_forward_pallas(x, weight, bias, input_state, BLOCK_SIZE_S)
    return y, (x, weight, bias, input_state)


def _depthwise_causal_convolution_pallas_bwd(BLOCK_SIZE_S: int, residuals: tuple, dy: jax.Array) -> tuple:
    x, weight, bias, input_state = residuals

    dx, dweight, dbias, dh0 = _depthwise_causal_convolution_backward_pallas(
        x, weight, input_state, dy, None, BLOCK_SIZE_S
    )

    return dx, dweight, (dbias if bias is not None else None), (dh0 if input_state is not None else None)


_depthwise_causal_convolution_pallas_op.defvjp(
    _depthwise_causal_convolution_pallas_fwd, _depthwise_causal_convolution_pallas_bwd
)


_BLOCK_SIZE_S = 128


def _depthwise_causal_convolution_pallas(
    x: jax.Array, weight: jax.Array, bias: jax.Array | None, input_state: jax.Array | None, output_state: bool
) -> tuple[jax.Array, jax.Array | None]:
    # hand-written VPU-only Pallas TPU kernel: a plain depthwise conv only reduces over the (typically tiny,
    # e.g. 4) kernel_size taps, so routing it through lax.conv's MXU/systolic-array path wastes it - this
    # kernel instead does the whole thing as unrolled elementwise shift-multiply-adds. See
    # pallas_implementation/{forward,backward}.py.
    K = weight.shape[-1]
    y = _depthwise_causal_convolution_pallas_op(x, weight, bias, input_state, _BLOCK_SIZE_S)
    final_state = _compute_final_state(x, input_state, K, output_state)

    return y, final_state


def depthwise_causal_convolution_jax(
    input: jax.Array,
    weight: jax.Array,
    bias: jax.Array | None = None,
    input_state: jax.Array | None = None,
    attention_mask: jax.Array | None = None,
    output_state: bool = False,
    activation_function: str | Callable[[jax.Array], jax.Array] | None = None,
    *,
    kernel_backend: KernelBackend | None = None,
) -> tuple[jax.Array, jax.Array | None]:
    """
    computes depthwise causal 1D convolution: `output[b, t, h] = act(bias[h] + sum_k weight[h, k] *
    z[b, t + k, h])` where `z` is `input` preceded by `kernel_size` raw history positions taken from
    `input_state` (or 0 if `input_state` is None), i.e. `output[b, t]` only depends on
    `input[b, t - kernel_size + 1 : t + 1]`.

    :param input: input tensor of shape (B, S, H)
    :type input: jax.Array
    :param weight: depthwise convolution weight of shape (H, K), K being the kernel size
    :type weight: jax.Array
    :param bias: bias tensor of shape (H,). None means no bias is added. Defaults to None.
    :type bias: jax.Array | None
    :param input_state: the `K` raw (pre-convolution) input positions preceding `input`, of shape (B, H, K).
        None is equivalent to a 0 tensor. Defaults to None.
    :type input_state: jax.Array | None
    :param attention_mask: mask of shape (B, S), zeroing out padding positions before and after the
        convolution. Defaults to None.
    :type attention_mask: jax.Array | None
    :param output_state: whether to also return the trailing `K` raw input positions (taken from `input`,
        falling back to `input_state` if `input` is shorter than `K`) for use as `input_state` in a
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
    :return: output tensor of shape (B, S, H), and the output state of shape (B, H, K) if `output_state` is
        True else None.
    :rtype: tuple[jax.Array, jax.Array | None]
    """

    assert input.ndim == 3
    B, _, H = input.shape

    assert weight.ndim == 2
    H_w, K = weight.shape
    assert H_w == H
    assert K >= 1

    if bias is not None:
        assert bias.shape == (H,)

    if input_state is not None:
        assert input_state.shape == (B, H, K)

    if activation_function is None or isinstance(activation_function, str):
        activation_function = _get_activation_function(activation_function)

    if kernel_backend is None:
        kernel_backend = Accelerator.get_kernel_backend()

    x = _apply_mask_to_padding_states(input, attention_mask)

    if kernel_backend == KernelBackend.pallas:
        output, final_state = _depthwise_causal_convolution_pallas(x, weight, bias, input_state, output_state)
    elif kernel_backend == KernelBackend.jax:
        output, final_state = _depthwise_causal_convolution_reference(x, weight, bias, input_state, output_state)
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    output = activation_function(output)
    output = _apply_mask_to_padding_states(output, attention_mask)

    return output, final_state
