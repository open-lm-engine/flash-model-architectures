import inspect
import os
from typing import Callable

import torch
import yaml
from torch.utils.cpp_extension import load as load_cpp_extension


CPP_MODULE_PREFIX = "cute_cuda_kernels"
CPP_BUILD_DIRECTORY = "build"
CPP_FUNCTIONS = {}
CPP_REGISTRY_YAML = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "cpp_registry.yml"), "r"))


@torch._dynamo.disable
def compile_cpp(name: str) -> None:
    function_map = []
    all_functions = []
    source_map = []
    build_directories = []
    for module in CPP_REGISTRY_YAML:
        function_map.append(module["functions"])
        all_functions.extend(module["functions"])
        source_map.append([os.path.join(os.path.dirname(__file__), source) for source in module["sources"]])
        build_directories.append(module["build_path"])

    assert len(all_functions) == len(set(all_functions)), "function names are not unique"

    # find which files the function belongs to
    for index, (functions, build_directory) in enumerate(zip(function_map, build_directories)):
        if name in functions:
            break

    full_build_path = os.path.join(CPP_BUILD_DIRECTORY, build_directory)
    os.makedirs(full_build_path, exist_ok=True)

    module = load_cpp_extension(
        f"{CPP_MODULE_PREFIX}_{build_directory}",
        sources=source_map[index],
        with_cuda=True,
        extra_cflags=[
            "-O3",
            "-Wall",
            "-shared",
            "-fPIC",
            "-fdiagnostics-color",
            f"-I{os.path.dirname(__file__)}",
        ],
        build_directory=full_build_path,
        verbose=True,
    )

    # populate all functions from the file
    for function in function_map[index]:
        CPP_FUNCTIONS[function] = getattr(module, function)


def get_cpp_function(name: str) -> Callable:
    function = CPP_FUNCTIONS.get(name, None)

    # if kernel is compiled, we return the torch op since its compatible with torch compile
    if function is None:
        compile_cpp(name)
        function = get_cpp_function(name)

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
