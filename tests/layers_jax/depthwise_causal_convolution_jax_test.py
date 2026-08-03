# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose


jax = pytest.importorskip("jax")

import haliax
import jax.numpy as jnp

from xma.layers_jax import DepthwiseCausalConvolutionJAX, depthwise_causal_convolution_jax


_TOLERANCE = {"atol": 1e-5, "rtol": 1e-5}


def _reference_numpy(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None,
    input_state: np.ndarray | None,
    output_state: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    # unifies prefill / decode / continuation exactly like `_depthwise_causal_convolution_reference`: pad the
    # raw input with a (possibly all-zero) K-wide state and slide a length-K window over the concatenation
    B, S, H = x.shape
    K = weight.shape[-1]

    xt = np.transpose(x, (0, 2, 1))
    state = np.zeros((B, H, K), dtype=np.float64) if input_state is None else input_state.astype(np.float64)
    full = np.concatenate([state, xt.astype(np.float64)], axis=-1)

    y = np.zeros((B, H, S), dtype=np.float64)
    for j in range(S):
        window = full[:, :, j + 1 : j + 1 + K]
        y[:, :, j] = (window * weight[None, :, :].astype(np.float64)).sum(-1)

    if bias is not None:
        y = y + bias[None, :, None].astype(np.float64)

    y = np.transpose(y, (0, 2, 1)).astype(x.dtype)
    final_state = full[:, :, -K:].astype(x.dtype) if output_state else None

    return y, final_state


def _generate_args() -> list:
    return list(
        product(
            [1, 2, 4],  # kernel_size
            [1, 2, 6, 9],  # sequence length: shorter than, equal to, or longer than kernel_size
            [False, True],  # has_input_state
            [False, True],  # add_bias
            [False, True],  # output_state
        )
    )


@pytest.mark.parametrize("kernel_size,S,has_input_state,add_bias,output_state", _generate_args())
def test_depthwise_causal_convolution_jax_forward(
    kernel_size: int, S: int, has_input_state: bool, add_bias: bool, output_state: bool
) -> None:
    B, H = 2, 5
    std = 0.1

    key_x, key_w, key_b, key_h0 = jax.random.split(jax.random.PRNGKey(0), 4)

    x = jax.random.normal(key_x, (B, S, H), dtype=jnp.float32) * std
    weight = jax.random.normal(key_w, (H, kernel_size), dtype=jnp.float32) * std
    bias = jax.random.normal(key_b, (H,), dtype=jnp.float32) * std if add_bias else None
    input_state = jax.random.normal(key_h0, (B, H, kernel_size), dtype=jnp.float32) * std if has_input_state else None

    y, final_state = depthwise_causal_convolution_jax(x, weight, bias, input_state, output_state=output_state)

    y_ref, final_state_ref = _reference_numpy(
        np.asarray(x),
        np.asarray(weight),
        np.asarray(bias) if bias is not None else None,
        np.asarray(input_state) if input_state is not None else None,
        output_state,
    )

    assert_allclose(np.asarray(y), y_ref, **_TOLERANCE)

    if output_state:
        assert final_state is not None
        assert final_state.shape == (B, H, kernel_size)
        assert_allclose(np.asarray(final_state), final_state_ref, **_TOLERANCE)
    else:
        assert final_state is None


@pytest.mark.parametrize("kernel_size", [1, 2, 4])
def test_depthwise_causal_convolution_jax_grad_runs(kernel_size: int) -> None:
    # `lax.conv_general_dilated` already ships well-tested autodiff rules in JAX itself, so this only checks
    # that gradients flow end to end with the right shapes/finiteness, not their numerical value (that's
    # covered by the pure jnp/lax.conv machinery upstream, not anything custom written here)
    B, S, H = 2, 6, 5
    std = 0.1

    key_x, key_w, key_b, key_h0 = jax.random.split(jax.random.PRNGKey(1), 4)

    x = jax.random.normal(key_x, (B, S, H), dtype=jnp.float32) * std
    weight = jax.random.normal(key_w, (H, kernel_size), dtype=jnp.float32) * std
    bias = jax.random.normal(key_b, (H,), dtype=jnp.float32) * std
    input_state = jax.random.normal(key_h0, (B, H, kernel_size), dtype=jnp.float32) * std

    def f(x, weight, bias, input_state):
        y, _ = depthwise_causal_convolution_jax(x, weight, bias, input_state, output_state=False)
        return y.sum()

    dx, dweight, dbias, dinput_state = jax.grad(f, argnums=(0, 1, 2, 3))(x, weight, bias, input_state)

    for name, grad, expected_shape in [
        ("dx", dx, x.shape),
        ("dweight", dweight, weight.shape),
        ("dbias", dbias, bias.shape),
        ("dinput_state", dinput_state, input_state.shape),
    ]:
        assert grad.shape == expected_shape, name
        assert bool(jnp.all(jnp.isfinite(grad))), name


@pytest.mark.parametrize("has_input_state", [False, True])
def test_depthwise_causal_convolution_module_works(has_input_state: bool) -> None:
    Embed = haliax.Axis("embed", 8)
    Batch = haliax.Axis("batch", 2)
    Pos = haliax.Axis("position", 6)
    kernel_size = 4

    key_init, key_input, key_state = jax.random.split(jax.random.PRNGKey(2), 3)

    module = DepthwiseCausalConvolutionJAX.init(
        Embed, kernel_size=kernel_size, activation_function="silu", add_bias=True, key=key_init
    )

    input = haliax.random.normal(key_input, (Batch, Pos, Embed))
    input_state = haliax.random.normal(key_state, (Batch, module.Embed, module.Kernel)) if has_input_state else None

    # this is a smoke test: it only checks that the module runs end to end and returns the expected shapes, not
    # that the output is numerically correct (that's covered at the op level by the tests above)
    output, output_state = module(input, input_state, output_state=True)

    assert output.axes == (Batch, Pos, Embed)
    assert output_state.axes == (Batch, module.Embed, module.Kernel)
