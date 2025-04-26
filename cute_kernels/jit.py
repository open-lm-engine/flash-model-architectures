import inspect
import os
from shutil import rmtree
from typing import Callable
from uuid import uuid4

import torch
from torch.utils.cpp_extension import load as load_cpp_extension


_CPP_MODULE_PREFIX = "cute_kernels"
_GLOBAL_RANK = int(os.getenv("RANK", 0))


@torch._dynamo.disable
def _get_cpp_function(function_name: str, source_files: list[str], build_directory: str) -> Callable:
    os.makedirs(build_directory, exist_ok=True)
    module_name = f"{_CPP_MODULE_PREFIX}_{function_name}"

    extra_cflags = ["-O3", "-Wall", "-shared", "-fPIC", "-fdiagnostics-color"]
    extra_cuda_cflags = ["-lineinfo"]
    extra_include_paths = [
        os.path.dirname(__file__),  # cute_kernels/include
        os.path.dirname(os.path.dirname(__file__)) + "/cutlass/include",  # cutlass
        os.path.dirname(os.path.dirname(__file__)) + "/cutlass/tools/util/include",  # cutlass
    ]

    is_distributed_enabled = torch.distributed.is_initialized()

    if is_distributed_enabled:
        if _GLOBAL_RANK == 0:
            module = load_cpp_extension(
                module_name,
                sources=source_files,
                with_cuda=True,
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                extra_include_paths=extra_include_paths,
                build_directory=build_directory,
                verbose=True,
            )

            torch.distributed.barrier()
        else:
            torch.distributed.barrier()

            module = load_cpp_extension(
                module_name,
                sources=source_files,
                with_cuda=True,
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                extra_include_paths=extra_include_paths,
                build_directory=build_directory,
                verbose=False,
            )

        torch.distributed.barrier()
    else:
        build_directory = os.path.join(build_directory, str(uuid4()))

        module = load_cpp_extension(
            module_name,
            sources=source_files,
            with_cuda=True,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_include_paths=extra_include_paths,
            build_directory=build_directory,
            verbose=True,
        )

    rmtree(build_directory, ignore_errors=True)

    return getattr(module, function_name)


def cpp_jit(
    function_name: str | None = None, extra_source_files: list[str] = [], build_directory: str | None = None
) -> Callable:
    cpp_function = None
    args_spec = None

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

        full_args = []
        full_args.extend(args)
        for variable_name in args_spec.args[len(args) :]:
            full_args.append(kwargs[variable_name])

        return cpp_function(*full_args)

    def _wrapper(function: Callable) -> Callable:
        nonlocal args_spec
        args_spec = inspect.getfullargspec(function)

        _run.__doc__ = function.__doc__
        _run.__name__ = function.__name__ if function_name is None else function_name
        _run.__signature__ = inspect.signature(function)

        return _run

    return _wrapper
