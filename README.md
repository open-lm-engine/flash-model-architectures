<!-- **************************************************
Copyright (c) 2025, Mayank Mishra
************************************************** -->

# Discord Server
Join the [discord server](https://discord.gg/AFDxmjH5RV) if you are interested in LLM architecture or distributed training/inference research.

# Efficient GPU kernels written in both CUDA and Triton

<p align="center">
  <img src="assets/cute-cat.webp" width="300px" height="300px">
</p>

## Modules

| Module   |  Triton   | CUDA |
|----------|-----------|------|
| [GRU](xma/modules/gru/__init__.py) | ✅ | ❌ |
| [MoE](xma/modules/moe/__init__.py) | ✅ | ✅ |
| [RNN](xma/modules/rnn/__init__.py) | ✅ | ❌ |

## Ops

| Module     | Triton | CUDA |
|------------|--------|------|
| [bmm](xma/ops/bmm/__init__.py)        | ✅     | ❌   |
| [continuous_count](xma/ops/continuous_count/__init__.py) | ❌ | ✅ |
| [cross_entropy](xma/ops/cross_entropy/__init__.py) | ✅ | ❌ |
| [fused_linear_cross_entropy](xma/ops/fused_linear_cross_entropy.py) | ✅ | ❌ |
| [fused_residual_add_rmsnorm](xma/ops/fused_residual_add_rmsnorm/__init__.py) | ✅ | ❌ |
| [gemm](xma/ops/gemm/__init__.py) | ✅ | ❌ |
| [grouped_gemm](xma/ops/grouped_gemm/__init__.py) | ❌ | ✅ |
| [rmsnorm](xma/ops/rmsnorm/__init__.py) | ✅ | ❌ |
| [pack_sequence](xma/ops/sequence_packing/__init__.py) | ✅ | ✅ |
| [softmax](xma/ops/softmax/__init__.py) | ✅ | ❌ |
| [swiglu](xma/ops/swiglu/__init__.py) | ✅ | ✅ |
| [swiglu_packed](xma/ops/swiglu/__init__.py) | ✅ | ❌ |
| [unpack_sequence](xma/ops/sequence_packing/__init__.py) | ✅ | ✅ |
