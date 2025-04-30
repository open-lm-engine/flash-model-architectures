#pragma once

#include <torch/extension.h>

#define AT_DISPATCH_CUSTOM_FLOAT_TYPES(TYPE, NAME, ...)                            \
    AT_DISPATCH_SWITCH(TYPE,                                                       \
                       NAME,                                                       \
                       AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)         \
                           AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__) \
                               AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__))

#define AT_DISPATCH_CUSTOM_INT_TYPES(TYPE, NAME, ...)                     \
    AT_DISPATCH_SWITCH(TYPE,                                              \
                       NAME,                                              \
                       AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__) \
                           AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__))
