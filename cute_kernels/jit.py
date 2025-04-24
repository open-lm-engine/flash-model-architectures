import importlib
import inspect
import os
from typing import Callable

import torch
import torch.distributed
from torch.utils.cpp_extension import load as load_cpp_extension


_CPP_MODULE_PREFIX = "cute_kernels"
_GLOBAL_RANK = int(os.getenv("RANK", 0))


@torch._dynamo.disable
def get_cpp_function(
    function_name: str, source_files: list[str] | None = None, build_directory: str | None = None
) -> Callable:
    os.makedirs(build_directory, exist_ok=True)
    module_name = f"{_CPP_MODULE_PREFIX}_{build_directory}"

    if _GLOBAL_RANK == 0:
        module = load_cpp_extension(
            module_name,
            sources=source_files,
            with_cuda=True,
            extra_cflags=[
                "-O3",
                "-Wall",
                "-shared",
                "-fPIC",
                "-fdiagnostics-color",
            ],
            extra_cuda_cflags=["-lineinfo"],
            extra_include_paths=[
                os.path.dirname(__file__),
                os.path.dirname(os.path.dirname(__file__)) + "/cutlass/include",
                os.path.dirname(os.path.dirname(__file__)) + "/cutlass/tools/util/include",
            ],
            build_directory=build_directory,
            verbose=True,
        )

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        module = importlib.import_module(module_name)

    return getattr(module, function_name)


def cpp_jit(function_name: str | None = None, source_files: list[str] | None = None) -> Callable:
    cpp_function = None
    args_spec = None

    def _run(*args, **kwargs):
        nonlocal cpp_function

        if cpp_function is None:
            cpp_function = get_cpp_function(_run.__name__, source_files)

        full_args = []
        full_args.extend(args)
        for variable_name in args_spec.args[len(args) :]:
            full_args.append(kwargs[variable_name])

        return cpp_function(*full_args)

    def _wrapper(function: Callable) -> Callable:
        _run.__doc__ = function.__doc__
        _run.__name__ = function.__name__ if function_name is None else function_name
        _run.__signature__ = inspect.signature(function)

        nonlocal args_spec
        args_spec = inspect.getfullargspec(function)

        return _run

    return _wrapper
