# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch


def get_max_seqlen_and_max_seqlen_tensor(
    max_seqlen: torch.Tensor | int | None,
) -> tuple[torch.Tensor | None, int | None]:
    if isinstance(max_seqlen, torch.Tensor):
        return max_seqlen, None
    else:
        return None, max_seqlen
