# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch


def get_num_elements_and_hidden_size(x: torch.Tensor) -> tuple[int]:
    hidden_size = x.size(-1)
    num_elements = x.numel() // hidden_size
    return num_elements, hidden_size
