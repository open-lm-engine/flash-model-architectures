# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class Experts_Torch(nn.Module):
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

        self.reset_parameters()

    def forward(
        self,
        input: torch.Tensor | tuple[torch.Tensor],
        expert_frequency: torch.Tensor,
        return_list: bool,
    ) -> torch.Tensor | list[torch.Tensor]:
        if isinstance(input, torch.Tensor):
            input = input.split(expert_frequency.tolist(), dim=0)

        input = [
            F.linear(input[i], self.weight[i], None if self.bias is None else self.bias[i])
            for i in range(self.num_experts)
        ]

        if not return_list:
            input = torch.cat(input)

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


class MoE_Torch(nn.Module):
    linear_class = Experts_Torch

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

        self.c_fc = self.linear_class(
            num_experts=num_experts,
            in_features=self.hidden_size,
            out_features=2 * self.intermediate_size if is_glu else self.intermediate_size,
            add_bias=add_bias,
            std=std,
        )

        self.act = activation_function

        self.c_proj = self.linear_class(
            num_experts=num_experts,
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            add_bias=add_bias,
            std=std,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape

        # hidden_states -> (batch_size, query_length, hidden_size)
        hidden_states = hidden_states.view(-1, self.hidden_size)
        # hidden_states -> (total_q, hidden_size)
        router_logits, router_weights, selected_experts = self._compute_routing_weights(hidden_states)

        # router_logits -> (total_q, num_experts)
        # router_weights -> (total_q, top_k)
        # selected_experts -> (total_q, top_k)

        hidden_states = self._compute_experts(hidden_states, router_weights, selected_experts)
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
        self, hidden_states: torch.Tensor, router_weights: torch.Tensor, selected_experts: torch.Tensor
    ) -> torch.Tensor:
        total_q = hidden_states.shape[0]

        # hidden_states -> (total_q, hidden_size)
        # router_weights -> (total_q, top_k)
        # selected_experts -> (total_q, top_k)

        fan_in_index, batch_gates, expert_frequency = self._compute_expert_assignment(router_weights, selected_experts)

        # fan_in_index -> (total_q * top_k)
        # batch_gates -> (total_q * top_k)
        # expert_frequency -> (num_experts)

        hidden_states = hidden_states[fan_in_index]

        # hidden_states -> (total_q * top_k, hidden_size)

        hidden_states = self.c_fc(hidden_states, expert_frequency, return_list=True)
        # hidden_states -> num_experts x (?, hidden_size)
        hidden_states = [self.act(i) for i in hidden_states]
        # hidden_states -> num_experts x (?, intermediate_size)
        hidden_states = self.c_proj(hidden_states, expert_frequency, return_list=False)
        # hidden_states -> (total_q * top_k, hidden_size)

        hidden_states = hidden_states * batch_gates.unsqueeze(-1)
        zeros = torch.zeros((total_q, self.hidden_size), dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = zeros.index_add(0, fan_in_index, hidden_states)

        # hidden_states -> (total_q, hidden_size)

        return hidden_states

    def _compute_expert_assignment(
        self, router_weights: torch.Tensor, selected_experts: torch.Tensor
    ) -> tuple[torch.Tensor]:
        # router_weights -> (total_q, top_k)
        # selected_experts -> (total_q, top_k)
        selected_experts = selected_experts.flatten()
        # selected_experts -> (total_q * top_k)

        expert_frequency = selected_experts.bincount(minlength=self.num_experts)
        # expert_frequency -> (num_experts)

        index_sorted_experts = selected_experts.argsort()
        # index_sorted_experts -> (total_q * top_k)
        fan_in_index = index_sorted_experts // self.top_k
        # fan_in_index -> (total_q * top_k)

        # gather the gate values for grouped input tokens
        router_weights = router_weights.flatten()
        # router_weights -> (total_q * top_k)
        batch_gates = router_weights[index_sorted_experts]
        # batch_gates -> (total_q * top_k)

        return fan_in_index, batch_gates, expert_frequency

    def _get_topk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.top_k == 1:
            x, indices = x.max(dim=-1, keepdim=True)
        else:
            x, indices = x.topk(self.top_k, dim=-1)

        return x, indices
