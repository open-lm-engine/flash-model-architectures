# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import yaml


YES = "✅"
NO = "❌"


kernels = yaml.full_load(open("tools/kernels.yml"))
backends = [("Triton", "triton"), ("CUDA", "cuda")]


def get_string(key: str) -> str:
    layers = ""
    for kernel, implementations in kernels[key].items():
        layers += f"| {kernel} |"
        for backend in backends:
            layers += f" {YES if backend[1] in implementations else NO} |"
        layers += "\n"

    return layers.rstrip()


readme = f"""<!-- **************************************************
Copyright (c) 2025, Mayank Mishra
************************************************** -->

# <img src="assets/xma.png" width="90px" height="30px" style="vertical-align: middle;"> (Accelerated Model Architectures)

XMA is a repository comprising of fast kernels for model training.  
We are planning on adding lots of experimental and fun model architectures with support for multiple accelerators like NVIDIA, AMD GPUs, Google TPUs and Amazon Trainiums.

## layers

| functional | {' | '.join([i[0] for i in backends])} |
|-|-|-|
{get_string('layers')}

## functional

| functional | {' | '.join([i[0] for i in backends])} |
|-|-|-|
{get_string('functional')}

# Discord Server
Join the [discord server](https://discord.gg/AFDxmjH5RV) if you are interested in LLM architecture or distributed training/inference research.
"""

open("README.md", "w").write(readme)
