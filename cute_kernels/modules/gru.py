# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn as nn

from ..kernels import gru_cute, gru_torch
from ..math import divide_if_divisible


class GRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        state_size: int,
        output_size: int,
        num_heads: int,
        add_bias: bool,
        gradient_clipping: float | None,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.gradient_clipping = gradient_clipping
        self.state_head_dim = divide_if_divisible(state_size, self.num_heads)

        self.input_projection = nn.Linear(input_size, 3 * state_size, bias=add_bias)
        self.state_weight = nn.Parameter(torch.empty(self.num_heads, self.state_head_dim, self.state_head_dim))
        self.forget_weight = nn.Parameter(torch.empty(self.num_heads, self.state_head_dim, self.state_head_dim))
        self.reset_weight = nn.Parameter(torch.empty(self.num_heads, self.state_head_dim, self.state_head_dim))
        self.output_projection = nn.Linear(state_size, output_size, bias=False)

        self.reset_parameters()

    def forward(
        self,
        input: torch.Tensor,
        input_state: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        use_kernel: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input = self.input_projection(input)
        input, forget_gate, reset_gate = input.chunk(3, dim=-1)

        input, forget_gate, reset_gate = [
            i.view(*input.size()[:-1], self.num_heads, self.state_head_dim) for i in (input, forget_gate, reset_gate)
        ]

        if input_state is not None:
            input_state = input_state.view(-1, self.num_heads, self.state_head_dim)

        input = (gru_cute if use_kernel else gru_torch)(
            input=input,
            weight=self.state_weight,
            forget_input=forget_gate,
            forget_weight=self.forget_weight,
            reset_input=reset_gate,
            reset_weight=self.reset_weight,
            input_state=input_state,
            gradient_clipping=self.gradient_clipping,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        del forget_gate, reset_gate

        if cu_seqlens is None:
            input_state = input[:, -1]
        else:
            input_state = input[cu_seqlens[1:] - 1]

        input_state = input_state.view(input_state.size(0), -1)

        input = input.view(*input.size()[:-2], -1)
        input = self.output_projection(input)

        return input, input_state

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.state_weight)
