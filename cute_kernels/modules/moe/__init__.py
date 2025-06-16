# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...kernel_backend import KernelBackend
from ...ops import continuous_count_cute
from .cuda_implementation import group_with_padding, grouped_gemm_experts_cute, ungroup_with_padding
from .triton_implementation import scattered_experts


class Experts(nn.Module):
    def __init__(
        self, num_experts: int, in_features: int, out_features: int, add_bias: bool = True, std: float | None = None
    ) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.empty(num_experts, out_features, in_features))

        self.bias = None
        if add_bias:
            self.bias = nn.Parameter(torch.empty(num_experts, out_features))

        self.std = std

        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer("N_array", torch.empty((num_experts,), dtype=torch.uint32))
        self.register_buffer("K_array", torch.empty((num_experts,), dtype=torch.uint32))

        self.reset_parameters()

    def forward(
        self,
        input: torch.Tensor,
        kernel_backend: KernelBackend,
        num_experts_per_token: int | None = None,
        expert_frequency: torch.Tensor | None = None,
        sorted_expert_idxs: torch.Tensor | None = None,
        sorted_scattered_idxs: torch.Tensor | None = None,
        expert_offsets: torch.Tensor | None = None,
        gates: torch.Tensor | None = None,
        grouped_in: bool = False,
        grouped_out: bool = False,
    ) -> torch.Tensor:
        if kernel_backend == KernelBackend.cuda:
            assert self.bias is None

            input = grouped_gemm_experts_cute(
                x=input, weight=self.weight, M_array=expert_frequency, N_array=self.N_array, K_array=self.K_array
            )
        elif kernel_backend == KernelBackend.triton:
            assert self.bias is None

            input = scattered_experts(
                inputs=input,
                expert_weights=self.weight.permute(0, 2, 1),
                k=num_experts_per_token,
                sorted_expert_idxs=sorted_expert_idxs,
                sorted_scattered_idxs=sorted_scattered_idxs,
                expert_offsets=expert_offsets,
                gates=gates,
                grouped_in=grouped_in,
                grouped_out=grouped_out,
            )
        elif kernel_backend == KernelBackend.torch:
            input = input.split(expert_frequency.tolist(), dim=0)
            input = [
                F.linear(input[i], self.weight[i], None if self.bias is None else self.bias[i])
                for i in range(self.num_experts)
            ]
            input = torch.cat(input, dim=0)
        else:
            raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

        return input

    def extra_repr(self):
        return "num_experts={}, in_features={}, out_features={}".format(
            self.num_experts, self.in_features, self.out_features
        )

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0, std=self.std)
        if hasattr(self, "bias") and self.bias is not None:
            self.bias.zero_()

        self.N_array.fill_(self.out_features)
        self.K_array.fill_(self.in_features)


class MoE(nn.Module):
    def __init__(
        self,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        intermediate_size: int,
        activation_function: Callable,
        is_glu: bool,
        add_bias: bool,
        std: float,
    ) -> None:
        super().__init__()

        self.num_experts = num_experts
        self.top_k = num_experts_per_tok

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate = nn.Linear(in_features=self.hidden_size, out_features=num_experts, bias=False)

        self.c_fc = Experts(
            num_experts=num_experts,
            in_features=self.hidden_size,
            out_features=2 * self.intermediate_size if is_glu else self.intermediate_size,
            add_bias=add_bias,
            std=std,
        )

        self.act = activation_function

        self.c_proj = Experts(
            num_experts=num_experts,
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            add_bias=add_bias,
            std=std,
        )

    def forward(
        self, hidden_states: torch.Tensor, kernel_backend: KernelBackend = KernelBackend.triton
    ) -> torch.Tensor:
        original_shape = hidden_states.shape

        # hidden_states -> (batch_size, query_length, hidden_size)
        hidden_states = hidden_states.view(-1, self.hidden_size)
        # hidden_states -> (total_q, hidden_size)
        router_logits, router_weights, selected_experts = self._compute_routing_weights(hidden_states)

        # router_logits -> (total_q, num_experts)
        # router_weights -> (total_q, top_k)
        # selected_experts -> (total_q, top_k)

        hidden_states = self._compute_experts(
            hidden_states, router_weights, selected_experts, kernel_backend=kernel_backend
        )

        hidden_states = hidden_states.view(original_shape)

        # hidden_states -> (batch_size, query_length, hidden_size)

        return hidden_states, router_logits

    def _compute_routing_weights(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
        # hidden_states -> (total_q, hidden_size)
        router_logits = self.gate(hidden_states)
        # router_logits -> (total_q, num_experts)

        router_weights, selected_experts = self._get_topk(router_logits)

        # router_weights -> (total_q, top_k)
        # selected_experts -> (total_q, top_k)

        router_weights = F.softmax(router_weights.float(), dim=-1)
        router_weights = router_weights.type_as(hidden_states)

        return router_logits, router_weights, selected_experts

    def _compute_experts(
        self,
        hidden_states: torch.Tensor,
        router_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        kernel_backend: KernelBackend,
    ) -> torch.Tensor:
        with torch.no_grad():
            sorted_expert_idxs, sorted_scattered_idxs = selected_experts.flatten().sort()
            expert_frequency = continuous_count_cute(
                sorted_expert_idxs, self.num_experts, kernel_backend=KernelBackend.cuda
            )

        T = hidden_states.size(0)

        if kernel_backend == KernelBackend.cuda:
            hidden_states, padded_expert_frequency, expert_padding_offset = group_with_padding(
                x=hidden_states,
                expert_frequency=expert_frequency,
                sorted_idxs=sorted_expert_idxs,
                scattered_idxs=sorted_scattered_idxs,
                top_k=self.top_k,
                pad_to_multiple_of=8,
            )

            hidden_states = self.c_fc(
                input=hidden_states, kernel_backend=kernel_backend, expert_frequency=padded_expert_frequency
            )
            hidden_states = self.act(hidden_states)
            hidden_states = self.c_proj(
                input=hidden_states, kernel_backend=kernel_backend, expert_frequency=padded_expert_frequency
            )

            hidden_states = ungroup_with_padding(
                x=hidden_states,
                expert_padding_offset=expert_padding_offset,
                sorted_idxs=sorted_expert_idxs,
                scattered_idxs=sorted_scattered_idxs,
                router_weights=router_weights,
                top_k=self.top_k,
                num_tokens=T,
                pad_to_multiple_of=8,
            )
        elif kernel_backend == KernelBackend.triton:
            with torch.no_grad():
                expert_offsets = expert_frequency.cumsum(-1)

            hidden_states = self.c_fc(
                input=hidden_states,
                kernel_backend=kernel_backend,
                num_experts_per_token=self.top_k,
                sorted_expert_idxs=sorted_expert_idxs,
                sorted_scattered_idxs=sorted_scattered_idxs,
                expert_offsets=expert_offsets,
                grouped_out=True,
            )
            hidden_states = self.act(hidden_states)
            hidden_states = self.c_proj(
                input=hidden_states,
                kernel_backend=kernel_backend,
                num_experts_per_token=1,
                sorted_expert_idxs=sorted_expert_idxs,
                sorted_scattered_idxs=sorted_scattered_idxs,
                expert_offsets=expert_offsets,
                grouped_in=True,
                gates=router_weights,
            )
        elif kernel_backend == KernelBackend.torch:
            # sort and group input tokens according to expert assignment
            fan_in_index = sorted_scattered_idxs // self.top_k

            # gather the gate values for grouped input tokens
            router_weights = router_weights.flatten()
            batch_gates = router_weights[sorted_scattered_idxs]

            hidden_states = hidden_states[fan_in_index]

            hidden_states = self.c_fc(
                input=hidden_states, kernel_backend=kernel_backend, expert_frequency=expert_frequency
            )
            hidden_states = self.act(hidden_states)
            hidden_states = self.c_proj(
                input=hidden_states, kernel_backend=kernel_backend, expert_frequency=expert_frequency
            )

            hidden_states = hidden_states * batch_gates.unsqueeze(-1)
            zeros = torch.zeros((T, self.hidden_size), dtype=hidden_states.dtype, device=hidden_states.device)
            hidden_states = zeros.index_add(0, fan_in_index, hidden_states)
        else:
            raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

        return hidden_states

    def _get_topk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.top_k == 1:
            x, indices = x.max(dim=-1, keepdim=True)
        else:
            x, indices = x.topk(self.top_k, dim=-1)

        return x, indices
