import torch


@torch.no_grad()
def continuous_count_and_sort_torch(x: torch.Tensor, size: int) -> tuple[torch.Tensor]:
    sorted_expert_idxs, sorted_scattered_idxs = x.flatten().sort()
    expert_offsets = sorted_expert_idxs.bincount(size=size).cumsum(-1)
    return sorted_expert_idxs, sorted_scattered_idxs, expert_offsets
