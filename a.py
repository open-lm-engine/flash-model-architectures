import textwrap
from pathlib import Path
from typing import Any, Sequence, Union

import torch
import torch._inductor.fx_passes.fuse_attention
from torch._inductor.fx_passes.joint_graph import patterns
from torch._inductor.pattern_matcher import (
    PatternExpr,
    PatternPrettyPrinter,
    SearchFn,
    TraceFn,
    _serialized_patterns,
    _TargetExpr,
    fwd_only,
    gen_pattern,
    init_once_fakemode,
    joint_fwd_bwd,
    register_replacement,
)
from torch._subclasses import FakeTensor


@torch.library.custom_op("test::mayank_op", mutates_args={"x"})
def _op(x: torch.Tensor, forward: bool) -> None:
    if forward:
        print("this is an OP's forward")
    else:
        print("this is an OP's backward")
    return


class _F(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        _op(x, True)
        return x

    @staticmethod
    def backward(ctx, x):
        _op(x, False)
        return x


def _sfdp_pattern_1(query, key, value):
    return torch.matmul(query, key.transpose(-2, -1)).softmax(dim=-1).matmul(value)


def _replacement_pattern_1(query, key, value):
    return _F.apply(query)
    # return query + key + value


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


@init_once_fakemode
def f():
    device = "cpu"
    q = torch.empty((2, 4, 8, 16), device=device, requires_grad=True)
    k = torch.empty((2, 4, 8, 16), device=device, requires_grad=True)
    v = torch.empty((2, 4, 8, 16), device=device, requires_grad=True)

    args = {
        "search_fn": _sfdp_pattern_1,
        "replace_fn": _replacement_pattern_1,
        "example_inputs": [q, k, v],
        "trace_fn": joint_fwd_bwd,
        "pass_dicts": patterns,
    }

    register_replacement(**args)

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

    register_replacement(**args)


f()


def g():
    _sfdp_pattern_1_compiled = torch.compile(_sfdp_pattern_1)

    device = "cpu"
    q = torch.empty((2, 4, 8, 16), device=device, requires_grad=True)
    k = torch.empty((2, 4, 8, 16), device=device, requires_grad=True)
    v = torch.empty((2, 4, 8, 16), device=device, requires_grad=True)

    o = _sfdp_pattern_1_compiled(q, k, v)
    o.sum().backward()


g()
