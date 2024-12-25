import torch


@torch.no_grad()
def contiguous_count_torch(x: torch.Tensor, size: int) -> torch.Tensor:
    return x.bincount(minlength=size).to(torch.uint32)
