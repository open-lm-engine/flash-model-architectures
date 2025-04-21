import torch


def pack_sequence_torch(x: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: torch.Tensor | int) -> torch.Tensor:
    batch_size = cu_seqlens.size(0) - 1
    other_dims = x.size()[1:]

    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    batch_indices = torch.arange(batch_size, device=x.device).repeat_interleave(seqlens)
    seq_indices = torch.cat([torch.arange(sl, device=x.device) for sl in seqlens])

    padded = torch.zeros(batch_size, max_seqlen, *other_dims, dtype=x.dtype, device=x.device)
    padded[batch_indices, seq_indices] = x

    return padded
