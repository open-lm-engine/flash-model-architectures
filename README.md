<!-- **************************************************
Copyright (c) 2025, Mayank Mishra
************************************************** -->

# Discord Server
Join the [discord server](https://discord.gg/AFDxmjH5RV) if you are interested in LLM architecture or distributed training/inference research.

# Efficient GPU kernels written in both CUDA and Triton

<img src="assets/xma.png" width="100px" height="100px">

## layers

| Module   |  Triton   | CUDA |
|----------|-----------|------|
| [GRU](xma/layers/gru.py) | ✅ | ❌ |
| [MoE](xma/layers/moe/__init__.py) | ✅ | ✅ |
| [RNN](xma/layers/rnn.py) | ✅ | ❌ |

## functional

| Module     | Triton | CUDA |
|------------|--------|------|
| [bmm](xma/functional/bmm/__init__.py)        | ✅     | ❌   |
| [continuous_count](xma/functional/continuous_count/__init__.py) | ❌ | ✅ |
| [cross_entropy](xma/functional/cross_entropy/__init__.py) | ✅ | ❌ |
| [fused_linear_cross_entropy](xma/functional/fused_linear_cross_entropy.py) | ✅ | ❌ |
| [fused_residual_add_rmsnorm](xma/functional/fused_residual_add_rmsnorm/__init__.py) | ✅ | ❌ |
| [grouped_gemm](xma/functional/grouped_gemm/__init__.py) | ❌ | ✅ |
| [rmsnorm](xma/functional/rmsnorm.py) | ✅ | ❌ |
| [pack_sequence](xma/functional/sequence_packing/__init__.py) | ✅ | ✅ |
| [softmax](xma/functional/softmax/__init__.py) | ✅ | ❌ |
| [swiglu](xma/functional/swiglu/__init__.py) | ✅ | ✅ |
| [swiglu_packed](xma/functional/swiglu/__init__.py) | ✅ | ❌ |
| [unpack_sequence](xma/functional/sequence_packing/__init__.py) | ✅ | ✅ |
