# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch.nn as nn


class XMAModule(nn.Module):
    def extra_repr(self) -> str:
        num_parameters = sum(p.numel() for p in self.parameters())
        return f"# parameters = {num_parameters:,}"
