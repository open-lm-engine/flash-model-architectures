# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from typing import Callable

import equinox as eqx
import haliax as hax
import jax
from haliax import Axis, NamedArray
from jaxtyping import PRNGKeyArray

from ...accelerator import KernelBackend
from .op import depthwise_causal_convolution_jax


class DepthwiseCausalConvolutionJAX(eqx.Module):
    weight: NamedArray
    bias: NamedArray | None

    Embed: Axis = eqx.field(static=True)
    Kernel: Axis = eqx.field(static=True)
    StateSize: Axis = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)
    activation_function: str | Callable[[jax.Array], jax.Array] | None = eqx.field(static=True)

    @staticmethod
    def init(
        Embed: Axis,
        kernel_size: int,
        activation_function: str | Callable[[jax.Array], jax.Array] | None,
        add_bias: bool,
        *,
        key: PRNGKeyArray,
    ) -> DepthwiseCausalConvolutionJAX:
        assert kernel_size > 1

        Kernel = Axis("kernel_size", kernel_size)
        StateSize = Axis("state_size", kernel_size - 1)

        weight_key, bias_key = jax.random.split(key, 2)
        bound = kernel_size**-0.5

        weight = hax.random.uniform(weight_key, (Embed, Kernel), minval=-bound, maxval=bound)
        bias = hax.random.uniform(bias_key, (Embed,), minval=-bound, maxval=bound) if add_bias else None

        return DepthwiseCausalConvolutionJAX(
            weight=weight,
            bias=bias,
            Embed=Embed,
            Kernel=Kernel,
            StateSize=StateSize,
            kernel_size=kernel_size,
            activation_function=activation_function,
        )

    def __call__(
        self,
        input: NamedArray,
        input_state: NamedArray | None = None,
        attention_mask: NamedArray | None = None,
        output_state: bool = False,
        *,
        kernel_backend: KernelBackend | None = None,
    ) -> tuple[NamedArray, NamedArray | None]:
        # input: (Batch, Pos, Embed); Batch and Pos are whatever's left over once Embed is accounted for
        Batch, Pos = [axis for axis in input.axes if axis != self.Embed]

        output, final_state = depthwise_causal_convolution_jax(
            input=input.rearrange((Batch, Pos, self.Embed)).array,
            weight=self.weight.rearrange((self.Embed, self.Kernel)).array,
            bias=self.bias.array if self.bias is not None else None,
            input_state=(
                input_state.rearrange((Batch, self.Embed, self.StateSize)).array if input_state is not None else None
            ),
            attention_mask=attention_mask.rearrange((Batch, Pos)).array if attention_mask is not None else None,
            output_state=output_state,
            activation_function=self.activation_function,
            kernel_backend=kernel_backend,
        )

        output = hax.named(output, (Batch, Pos, self.Embed))
        final_state = hax.named(final_state, (Batch, self.Embed, self.StateSize)) if final_state is not None else None

        return output, final_state
