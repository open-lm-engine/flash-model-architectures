# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch


def init_inductor(cache_size_limit: int) -> None:
    torch._dynamo.config.cache_size_limit = cache_size_limit
    torch._dynamo.config.accumulated_cache_size_limit = 1024
