import importlib
import inspect
import os
from typing import Callable

import torch
import torch.distributed
import yaml
from torch.utils.cpp_extension import load as load_cpp_extension

from .utils import get_boolean_env_variable


_CPP_MODULE_PREFIX = "cute_kernels"
_CPP_BUILD_DIRECTORY = "build"
_CPP_FUNCTIONS = {}
_CPP_REGISTRY_YAML = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "cpp_registry.yml"), "r"))
_BUILD_ON_EVERY_NODE = get_boolean_env_variable("BUILD_ON_EVERY_NODE", True)


@torch._dynamo.disable
def get_cpp_function(name: str) -> Callable:
    function = _CPP_FUNCTIONS.get(name, None)

    # if kernel is compiled, we return the torch op since its compatible with torch compile
    if function is None:
        function_map = []
        all_functions = []
        source_map = []
        build_directories = []
        for module in _CPP_REGISTRY_YAML:
            function_map.append(module["functions"])
            all_functions.extend(module["functions"])
            source_map.append([os.path.join(os.path.dirname(__file__), source) for source in module["sources"]])
            build_directories.append(module["build_path"])

        assert len(all_functions) == len(set(all_functions)), "function names are not unique"

        # find which files the function belongs to
        for index, (functions, build_directory) in enumerate(zip(function_map, build_directories)):
            if name in functions:
                break

        full_build_path = os.path.join(_CPP_BUILD_DIRECTORY, build_directory)
        os.makedirs(full_build_path, exist_ok=True)

        module_name = f"{_CPP_MODULE_PREFIX}_{build_directory}"

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            module = load_cpp_extension(
                module_name,
                sources=source_map[index],
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
                build_directory=full_build_path,
                verbose=True,
            )

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            module = importlib.import_module(module_name)

        # populate all functions from the file
        for function in function_map[index]:
            _CPP_FUNCTIONS[function] = getattr(module, function)

        function = _CPP_FUNCTIONS[name]

    return function


def cpp_jit(function_name: str) -> Callable:
    cpp_function = None
    args_spec = None

    def _run(*args, **kwargs):
        nonlocal cpp_function

        if cpp_function is None:
            cpp_function = get_cpp_function(function_name)

        full_args = []
        full_args.extend(args)
        for variable_name in args_spec.args[len(args) :]:
            full_args.append(kwargs[variable_name])

        return cpp_function(*full_args)

    def inner(function: Callable) -> Callable:
        _run.__signature__ = inspect.signature(function)
        _run.__name__ = function.__name__

        nonlocal args_spec
        args_spec = inspect.getfullargspec(function)

        return _run

    return inner
