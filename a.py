import functools
import importlib
import os
import textwrap
from math import log
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, Union

import torch
import torch._inductor.fx_passes.fuse_attention
import torch.utils._pytree as pytree
from torch._inductor.fx_passes.joint_graph import patterns
from torch._inductor.pattern_matcher import (
    FakeTensorMode,
    Match,
    PatternExpr,
    PatternPrettyPrinter,
    ReplaceFn,
    SearchFn,
    TraceFn,
    _known_precompiled_patterns,
    _PassDictsType,
    _return_true,
    _serialized_patterns,
    _TargetExpr,
    fwd_only,
    gen_pattern,
    init_once_fakemode,
    joint_fwd_bwd,
    register_replacement,
    unset_fake_temporarily,
)
from torch._subclasses import FakeTensor


@torch.library.custom_op("test::mayank_op", mutates_args={"x"})
def _op(x: torch.Tensor) -> None:
    return


class _F(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x
        # return _op(x)

    @staticmethod
    def backward(ctx, x):
        return x
        # return _op(x)


def _sfdp_pattern_1(query, key, value):
    return torch.matmul(query, key.transpose(-2, -1)).softmax(dim=-1).matmul(value)


def _replacement_pattern_1(query, key, value):
    print("hi")
    # return _F.apply(q)
    return query + key + value


SERIALIZED_PATTERN_PATH = Path("./")


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
            try:
                if isinstance(attr, type) and issubclass(attr, (PatternExpr, _TargetExpr)):
                    pattern_matcher_imports.append(name)
            except TypeError:
                pass

        formatted_imports = ",\n   ".join(pattern_matcher_imports)
        formatted_imports = f"from torch._inductor.pattern_matcher import (\n   {formatted_imports},\n)\n"
        return f"{file_template}{formatted_imports}"

    if not SERIALIZED_PATTERN_PATH.is_dir():
        raise RuntimeError(f"Could not find serialized patterns directory at {SERIALIZED_PATTERN_PATH}")

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

    with open(SERIALIZED_PATTERN_PATH / f"{pattern_name}.py", write_mode) as f:
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
    build: bool = True,
) -> None:
    # Make sure the example_inputs is materialized.
    example_inputs = tuple(example_inputs)

    if build:
        pat = _serialize_pattern(unique_name, search_fn, example_inputs, trace_fn, scalar_workaround)
    else:
        pattern_name = search_fn.__name__
        m = importlib.import_module(f"_sfdp_pattern_1")
        if not m or not hasattr(m, unique_name):
            assert False
        pat = getattr(m, unique_name)

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


@init_once_fakemode
def f():
    name = "mayank"

    device = "cpu"
    q = torch.empty((2, 4, 8, 16), device=device, requires_grad=True)
    k = torch.empty((2, 4, 8, 16), device=device, requires_grad=True)
    v = torch.empty((2, 4, 8, 16), device=device, requires_grad=True)

    training_name = name + "_training"
    args = {
        "search_fn": _sfdp_pattern_1,
        "replace_fn": _replacement_pattern_1,
        "example_inputs": [q, k, v],
        "trace_fn": joint_fwd_bwd,
        "pass_dicts": patterns,
    }

    gen_register_replacement(training_name, **args)

    inference_name = name + "_inference"
    args = {
        "search_fn": _sfdp_pattern_1,
        "replace_fn": _replacement_pattern_1,
        "example_inputs": [q, k, v],
        "trace_fn": fwd_only,
        "pass_dicts": patterns,
        # with dropout turned into clone, we end up with a number of
        # semantically identical graphs
        "skip_duplicates": True,
    }

    gen_register_replacement(inference_name, **args)


f()


def g():
    _sfdp_pattern_1_compiled = torch.compile(_sfdp_pattern_1)

    device = "cpu"
    q = torch.empty((2, 4, 8, 16), device=device, requires_grad=True)
    k = torch.empty((2, 4, 8, 16), device=device, requires_grad=True)
    v = torch.empty((2, 4, 8, 16), device=device, requires_grad=True)

    o = _sfdp_pattern_1_compiled(q, k, v)

    print(o)
    print(torch._dynamo.utils.counters)


g()
