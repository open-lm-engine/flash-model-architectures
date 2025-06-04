# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from .torch_implementation import Experts_Torch, MoE_Torch
from .triton_implementation import bincount, scattered_experts


class Experts_Cute(Experts_Torch):
    def forward(
        self,
        hidden_states: torch.Tensor,
        k: int,
        sorted_expert_idxs: torch.Tensor,
        sorted_scattered_idxs: torch.Tensor,
        expert_offsets: torch.Tensor,
        gates: torch.Tensor | None = None,
        grouped_in: bool = False,
        grouped_out: bool = False,
    ) -> torch.Tensor:
        hidden_states = scattered_experts(
            hidden_states,
            self.weight.permute(0, 2, 1),
            k,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            gates,
            grouped_in,
            grouped_out,
        )

        return hidden_states


class MoE_Cute(MoE_Torch):
    linear_class = Experts_Cute

    def _compute_experts(
        self, hidden_states: torch.Tensor, router_weights: torch.Tensor, selected_experts: torch.Tensor
    ) -> torch.Tensor:
        sorted_expert_idxs, sorted_scattered_idxs = selected_experts.flatten().sort()
        expert_offsets = bincount(sorted_expert_idxs, self.num_experts).cumsum(-1)

        hidden_states = self.c_fc(
            hidden_states,
            self.top_k,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            grouped_out=True,
        )
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(
            hidden_states,
            1,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            grouped_in=True,
            gates=router_weights,
        )
        return hidden_states
