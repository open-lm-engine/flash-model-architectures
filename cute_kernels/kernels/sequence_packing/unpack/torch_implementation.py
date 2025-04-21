import torch


def unpack_sequence(x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    batch_size = cu_seqlens.size(0) - 1
    other_dims = x.shape[2:]

    sum_seqlens = cu_seqlens[-1]
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    batch_indices = torch.arange(batch_size, device=x.device).repeat_interleave(seqlens)
    seq_indices = torch.cat([torch.arange(sl, device=x.device) for sl in seqlens])

    unpadded = torch.zeros(sum_seqlens, *other_dims, dtype=x.dtype, device=x.device)
    unpadded = x[batch_indices, seq_indices]

    return unpadded
