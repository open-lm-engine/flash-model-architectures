# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch.nn as nn


class XMAModule(nn.Module):
    def extra_repr(self) -> str:
        num_parameters = 0
        for p in self.parameters():
            num_parameters += p.numel()

        return f"# parameters = {num_parameters:,}"
