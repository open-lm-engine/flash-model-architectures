import torch


def gemm_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    alpha: float = 1,
    beta: float = 1,
    is_a_transposed: bool = False,
    is_b_transposed: bool = False,
) -> torch.Tensor:
    if is_a_transposed:
        a = a.T

    if is_b_transposed:
        b = b.T

    return torch.addmm(c, a, b, alpha=alpha, beta=beta)
