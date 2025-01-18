import torch


def gemm_torch(
    a: torch.Tensor, b: torch.Tensor, is_a_transposed: bool = False, is_b_transposed: bool = False
) -> torch.Tensor:
    if is_a_transposed:
        a = a.T

    if is_b_transposed:
        b = b.T

    return a @ b
