import importlib
import os
import textwrap
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, Union

import torch
import torch.utils._pytree as pytree
from torch._inductor.pattern_matcher import (
    Match,
    PatternExpr,
    PatternPrettyPrinter,
    ReplaceFn,
    SearchFn,
    TraceFn,
    _known_precompiled_patterns,
    _PassDictsType,
    _return_true,
    _serialize_pattern,
    _serialized_patterns,
    _TargetExpr,
    gen_pattern,
    register_replacement,
)
from torch._subclasses import FakeTensor


_CACHE_DIRECTORY = Path(os.path.dirname(__file__)) / "graphs"
_IMPORT_CACHE = ".cute_inductor.graphs"


def _serialize_pattern(
    unique_name: str,
    search_fn: SearchFn,
    example_inputs: Sequence[Any],
    trace_fn: TraceFn,
    scalar_workaround: Union[dict[str, Union[float, int]], None],
) -> PatternExpr:
    def get_file_template() -> str:
        auto_generated_msg = textwrap.dedent(
            """\
            # This is an auto-generated file. Please do not modify it by hand.
            # To re-generate, run:
            # cd ~/pytorch && python torchgen/fuse/gen_patterns.py
            """
        )

        file_template = textwrap.dedent(
            """\
            # mypy: ignore-errors

            # noqa: F401, E501
            {msg}
            import torch
            import torch._inductor
            import operator

            aten = torch.ops.aten
            prims = torch.ops.prims

            """
        ).format(msg=auto_generated_msg)

        pattern_matcher_imports = []
        for name in dir(torch._inductor.pattern_matcher):
            attr = getattr(torch._inductor.pattern_matcher, name)
            if isinstance(attr, type) and issubclass(attr, (PatternExpr, _TargetExpr)):
                pattern_matcher_imports.append(name)

        formatted_imports = ",\n   ".join(pattern_matcher_imports)
        formatted_imports = f"from torch._inductor.pattern_matcher import (\n   {formatted_imports},\n)\n"
        return f"{file_template}{formatted_imports}"

    if not _CACHE_DIRECTORY.is_dir():
        raise RuntimeError(f"Could not find serialized patterns directory at {_CACHE_DIRECTORY}")

    pattern_name = search_fn.__name__

    from torch._functorch import config as functorch_config

    with functorch_config.patch(functionalize_rng_ops=False):
        pattern = gen_pattern(search_fn, example_inputs, trace_fn, scalar_workaround)

    serialized_pattern = PatternPrettyPrinter.run(pattern, output_name=unique_name)
    if pattern_name not in _serialized_patterns:
        write_mode = "w"
        _serialized_patterns.add(pattern_name)
    else:
        write_mode = "a"

    file_template = get_file_template()

    with open(_CACHE_DIRECTORY / f"{pattern_name}.py", write_mode) as f:
        if write_mode == "w":
            f.write(file_template)
        else:
            f.write("\n\n")
        f.write(serialized_pattern)
        f.write("\n")

    return pattern


def gen_register_replacement(
    unique_name: str,
    search_fn: SearchFn,
    replace_fn: ReplaceFn,
    example_inputs: Iterable[Any],
    trace_fn: TraceFn,
    pass_dicts: Union[_PassDictsType, Sequence[_PassDictsType]],
    extra_check: Callable[[Match], bool] = _return_true,
    scalar_workaround: Union[dict[str, Union[float, int]], None] = None,
    exclusive_arg_names: Sequence[str] = (),
    skip_duplicates: bool = False,
) -> None:
    # Make sure the example_inputs is materialized.
    example_inputs = tuple(example_inputs)

    pattern_name = search_fn.__name__

    os.makedirs(_IMPORT_CACHE.split(".")[1], exist_ok=True)
    m = importlib.import_module(f"{_IMPORT_CACHE}.{pattern_name}", package="cute_kernels")

    if not m or not hasattr(m, unique_name):
        pat = _serialize_pattern(unique_name, search_fn, example_inputs, trace_fn, scalar_workaround)
    else:
        pat = getattr(m, unique_name, None)

    if pat is None:
        pat = _serialize_pattern(unique_name, search_fn, example_inputs, trace_fn, scalar_workaround)

    for arg in pytree.tree_iter(example_inputs):
        if isinstance(arg, FakeTensor) and arg.constant is not None:
            # This can be a problem - small fake tensors (e.g. `tensor(2)`) will
            # hold onto their original constant value - and by stashing it here
            # will cause a memory leak if the constant value is on GPU.
            # Since this is just an optimization we can clear it out.
            arg.constant = None

    _known_precompiled_patterns.append((search_fn, example_inputs, trace_fn, scalar_workaround, pat))
    register_replacement(
        search_fn,
        replace_fn,
        example_inputs,
        trace_fn,
        pass_dicts,
        extra_check,
        scalar_workaround,
        exclusive_arg_names,
        search_fn_pattern=pat,
        skip_duplicates=skip_duplicates,
    )
