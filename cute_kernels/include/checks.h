// **************************************************
// Copyright (c) 2025, Mayank Mishra
// **************************************************

#pragma once

#define CHECK_CUDA_TENSOR(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS_TENSOR(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CONTIGUOUS_CUDA_TENSOR(x) \
    CHECK_CUDA_DEVICE(x);               \
    CHECK_CONTIGUOUS(x)
