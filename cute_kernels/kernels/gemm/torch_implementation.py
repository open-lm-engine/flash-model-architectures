import torch


def gemm_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b
