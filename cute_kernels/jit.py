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
def _get_cpp_function(function_name: str, source_files: list[str], build_directory: str) -> Callable:
    os.makedirs(build_directory, exist_ok=True)
    module_name = f"{_CPP_MODULE_PREFIX}_{function_name}"

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
                os.path.dirname(__file__),  # cute_kernels/include
                os.path.dirname(os.path.dirname(__file__)) + "/cutlass/include",  # cutlass
                os.path.dirname(os.path.dirname(__file__)) + "/cutlass/tools/util/include",  # cutlass
            ],
            build_directory=build_directory,
            verbose=True,
        )

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        module = importlib.import_module(module_name)

    return getattr(module, function_name)


def cpp_jit(
    function_name: str | None = None, extra_source_files: list[str] = [], build_directory: str | None = None
) -> Callable:
    cpp_function = None

    source_files = []
    source_files.extend(extra_source_files)

    calling_filename = inspect.stack()[1].filename
    calling_directory = os.path.dirname(calling_filename)

    for dirname, _, filenames in os.walk(calling_directory):
        filenames = [os.path.join(dirname, f) for f in filenames]
        filenames = filter(lambda f: os.path.splitext(f)[1] in [".cu", ".cpp"], filenames)
        source_files.extend(filenames)

    if build_directory is None:
        build_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), "build")

    def _run(*args, **kwargs):
        nonlocal cpp_function

        if cpp_function is None:
            cpp_function = _get_cpp_function(_run.__name__, source_files, build_directory)

        return cpp_function(*args, **kwargs)

    def _wrapper(function: Callable) -> Callable:
        _run.__doc__ = function.__doc__
        _run.__name__ = function.__name__ if function_name is None else function_name
        _run.__signature__ = inspect.signature(function)

        return _run

    return _wrapper
