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

from xma import KernelBackend
from xma.layers_jax import DepthwiseCausalConvolutionJAX, depthwise_causal_convolution_jax


_TOLERANCE = {"atol": 2e-4, "rtol": 0}


def _reference_numpy(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None,
    input_state: np.ndarray | None,
    output_state: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    B, S, H = x.shape
    K = weight.shape[-1]

    xt = np.transpose(x, (0, 2, 1))
    state = np.zeros((B, H, K), dtype=np.float32) if input_state is None else input_state.astype(np.float32)
    full = np.concatenate([state, xt.astype(np.float32)], axis=-1)

    y = np.zeros((B, H, S), dtype=np.float32)
    for j in range(S):
        window = full[:, :, j + 1 : j + 1 + K]
        y[:, :, j] = (window * weight[None, :, :].astype(np.float32)).sum(-1)

    if bias is not None:
        y = y + bias[None, :, None].astype(np.float32)

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

    y, final_state = depthwise_causal_convolution_jax(
        x, weight, bias, input_state, output_state=output_state, kernel_backend=KernelBackend.jax
    )

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
    B, S, H = 2, 6, 5
    std = 0.1

    key_x, key_w, key_b, key_h0 = jax.random.split(jax.random.PRNGKey(1), 4)

    x = jax.random.normal(key_x, (B, S, H), dtype=jnp.float32) * std
    weight = jax.random.normal(key_w, (H, kernel_size), dtype=jnp.float32) * std
    bias = jax.random.normal(key_b, (H,), dtype=jnp.float32) * std
    input_state = jax.random.normal(key_h0, (B, H, kernel_size), dtype=jnp.float32) * std

    def f(x, weight, bias, input_state):
        y, _ = depthwise_causal_convolution_jax(
            x, weight, bias, input_state, output_state=False, kernel_backend=KernelBackend.jax
        )

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

    output, output_state = module(input, input_state, output_state=True, kernel_backend=KernelBackend.jax)

    assert output.axes == (Batch, Pos, Embed)
    assert output_state.axes == (Batch, module.Embed, module.Kernel)


def _generate_pallas_args() -> list:
    return list(
        product(
            [2, 4],  # kernel_size: the pallas kernel_backend assumes kernel_size > 1
            [3, 16, 37, 130],  # sequence length: shorter than, equal to, or not a multiple of the internal block size
            [False, True],  # has_input_state
            [False, True],  # add_bias
        )
    )


@pytest.mark.parametrize("kernel_size,S,has_input_state,add_bias", _generate_pallas_args())
def test_depthwise_causal_convolution_pallas(kernel_size: int, S: int, has_input_state: bool, add_bias: bool) -> None:
    if jax.default_backend() != "tpu":
        pytest.skip("KernelBackend.pallas is only supported on TPU")

    B, H = 2, 5
    std = 0.1

    key_x, key_w, key_b, key_h0, key_dy = jax.random.split(jax.random.PRNGKey(3), 5)

    x = jax.random.normal(key_x, (B, S, H), dtype=jnp.float32) * std
    weight = jax.random.normal(key_w, (H, kernel_size), dtype=jnp.float32) * std
    bias = jax.random.normal(key_b, (H,), dtype=jnp.float32) * std if add_bias else None
    input_state = jax.random.normal(key_h0, (B, H, kernel_size), dtype=jnp.float32) * std if has_input_state else None

    def _run(kernel_backend: KernelBackend, x: jax.Array, weight: jax.Array, bias, input_state):
        return depthwise_causal_convolution_jax(
            x, weight, bias, input_state, output_state=False, kernel_backend=kernel_backend
        )

    (y_kernel, _), vjp_kernel = jax.vjp(
        lambda x, weight, bias, input_state: _run(KernelBackend.pallas, x, weight, bias, input_state),
        x,
        weight,
        bias,
        input_state,
    )

    (y_expected, _), vjp_expected = jax.vjp(
        lambda x, weight, bias, input_state: _run(KernelBackend.jax, x, weight, bias, input_state),
        x,
        weight,
        bias,
        input_state,
    )

    assert_allclose(np.asarray(y_kernel), np.asarray(y_expected), **_TOLERANCE)

    # dy = jax.random.normal(key_dy, y_kernel.shape, dtype=jnp.float32) * std

    # dx_kernel, dweight_kernel, dbias_kernel, dinput_state_kernel = vjp_kernel((dy, None))
    # dx_expected, dweight_expected, dbias_expected, dinput_state_expected = vjp_expected((dy, None))

    # assert_allclose(np.asarray(dx_kernel), np.asarray(dx_expected), **_TOLERANCE)
    # assert_allclose(np.asarray(dweight_kernel), np.asarray(dweight_expected), **_TOLERANCE)

    # if add_bias:
    #     assert_allclose(np.asarray(dbias_kernel), np.asarray(dbias_expected), **_TOLERANCE)
    # else:
    #     assert dbias_kernel is None
    #     assert dbias_expected is None

    # if has_input_state:
    #     assert_allclose(np.asarray(dinput_state_kernel), np.asarray(dinput_state_expected), **_TOLERANCE)
    # else:
    #     assert dinput_state_kernel is None
    #     assert dinput_state_expected is None
