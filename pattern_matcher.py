import inspect
import itertools
from collections.abc import Iterable, Sequence
from typing import Any, Callable, Union

import torch
import torch._guards
import torch.fx
from torch._functorch import config as functorch_config
from torch._functorch.aot_autograd import aot_function, make_boxed_func
from torch._inductor.fx_passes.joint_graph import pointless_view
from torch._inductor.fx_passes.post_grad import remove_noop_ops
from torch._inductor.pattern_matcher import (
    CallFunction,
    GraphPatternEntry,
    KeywordArg,
    Match,
    PatternExpr,
    PatternMatcherPass,
    ReplaceFn,
    ReplacementPatternEntry,
    SearchFn,
    TraceFn,
    _PassDictsType,
    _return_true,
    _transfer_meta,
    check_and_add_duplicate_pattern,
    clone_graph,
    default_partition,
    enable_python_dispatcher,
    fwd_only,
    fx_to_pattern,
    gen_pattern_and_search_gm,
    is_match,
    joint_fwd_bwd,
    make_fx,
    select_decomp_table,
)
from torch._prims_common import is_integer_dtype
from torch.fx.experimental.symbolic_shapes import guard_size_oblivious


@torch.no_grad()
def fwd_only(
    fn: Callable[..., Any],
    args: Sequence[Any],
    *,
    run_functional_passes: bool = True,
    get_decomp_fn: Callable[..., Any] | None = None,
) -> torch.fx.GraphModule:
    """Build a normalized inference graph, for use with fx_to_pattern"""
    # TODO - look into using aot autograd, asserting no mutating ops here
    with enable_python_dispatcher():
        decompositions = get_decomp_fn() if get_decomp_fn is not None else select_decomp_table()
        gm = make_fx(fn, decompositions, tracing_mode="real")(*args)

    if run_functional_passes:
        remove_noop_ops(gm.graph)
        gm.graph.eliminate_dead_code()

    gm.recompile()
    return gm


@torch.enable_grad()
def joint_fwd_bwd(fn: Callable[..., Any], args: Sequence[Any]) -> torch.fx.GraphModule:
    """Build a normalized training graph, for use with fx_to_pattern"""
    gm: torch.fx.GraphModule | None = None

    def record_joint_graph(
        joint_graph: torch.fx.GraphModule, inputs: Sequence[Any], **kwargs: Any
    ) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
        nonlocal gm
        assert not gm
        gm = clone_graph(joint_graph)
        return default_partition(joint_graph, inputs, **kwargs)

    with torch._guards.tracing(None):
        aot_function(
            fn,
            lambda g, i: make_boxed_func(g),
            partition_fn=record_joint_graph,
            decompositions=select_decomp_table(),
            keep_inference_input_mutations=True,
            enable_log=False,
        )(*args)
    assert gm

    remove_noop_ops(gm.graph)

    matcher_pass = PatternMatcherPass()

    pattern = CallFunction(torch.ops.aten.view.default, KeywordArg("arg"), KeywordArg("size"))
    GraphPatternEntry(pattern=pattern, handler=pointless_view, extra_check=_return_true).register(
        matcher_pass.patterns
    )
    matcher_pass.apply(gm.graph)  # type: ignore[arg-type]

    # remove in/out specs
    gm.graph._codegen = torch.fx.graph.CodeGen()
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm


@functorch_config.patch(functionalize_rng_ops=False)
def gen_pattern_and_search_gm(
    search_fn: SearchFn,
    example_inputs: Sequence[Any],
    trace_fn: TraceFn,
    scalar_workaround: Union[dict[str, Union[float, int]], None] = None,
    exclusive_arg_names: Sequence[str] = (),
) -> tuple[PatternExpr, torch.fx.GraphModule]:
    argnames = [*inspect.signature(search_fn).parameters.keys()]

    if scalar_workaround is None:
        scalar_workaround = {}
    flat_inputs = []
    input_idx = 0  # Positional arguments index

    for argname in argnames:
        if argname in scalar_workaround:
            flat_inputs.append(scalar_workaround[argname])
        else:
            flat_inputs.append(example_inputs[input_idx])
            input_idx += 1

    search_gm = trace_fn(search_fn, flat_inputs)
    return (
        fx_to_pattern(
            search_gm,
            ignore_types=(int, float, list, torch.device, torch.dtype),
            argnames=argnames,
            scalar_workaround=scalar_workaround,
            exclusive_arg_names=exclusive_arg_names,
        ),
        search_gm,
    )


def register_replacement(
    search_fn: SearchFn,
    replace_fn: ReplaceFn,
    example_inputs: Iterable[Any],
    trace_fn: TraceFn,
    pass_dicts: Union[_PassDictsType, Sequence[_PassDictsType]],
    extra_check: Callable[[Match], bool] = _return_true,
    scalar_workaround: Union[dict[str, Union[float, int]], None] = None,
    exclusive_arg_names: Sequence[str] = (),
    skip_duplicates: bool = False,
) -> bool:
    argnames_static = [*inspect.signature(search_fn).parameters.keys()]

    def check_fn(match: Match) -> bool:
        argnames = list(argnames_static)
        for name in argnames:
            if name not in match.kwargs:
                raise RuntimeError(
                    f"Not all inputs to pattern found in match.kwargs. Perhaps one "
                    f"of the inputs is unused? argnames={argnames}, match.kwargs={match.kwargs}"
                )

        args = list(
            torch.fx.map_arg(  # type: ignore[arg-type]
                [match.kwargs[name] for name in argnames], lambda n: n.meta["val"]
            )
        )
        sym_args: list[torch.SymInt] = []
        with torch._dynamo.utils.detect_fake_mode(args):
            for i, grad in enumerate(requires_grad):
                if isinstance(args[i], torch.Tensor):
                    if grad and is_integer_dtype(args[i].dtype):
                        return False

                    args[i] = torch.empty_strided(
                        args[i].size(),
                        args[i].stride(),
                        dtype=args[i].dtype,
                        device=args[i].device,
                        requires_grad=grad,
                    )
                    for v in itertools.chain(args[i].shape, args[i].stride()):
                        if isinstance(v, torch.SymInt) and all(guard_size_oblivious(v != a) for a in sym_args):
                            sym_args.append(v)

            # If we were given a pre-traced pattern then use that instead of
            # retracing. Note that this means the pattern has to be independent
            # of its args.
            if sym_args:
                # AOT Autograd and make fx will dedupe symbolic shape size
                # accesses of sym ints that appear as inputs
                # We don't want the sym_size uses to interfere with pattern matching
                # so we provide them as inputs.
                # Later, when we actually do the replacement, the symbolic shape
                # sizes will get re-traced and added to the graph.

                def search_fn_new(*args_new: Any) -> Any:
                    return search_fn(*args_new[len(args_new) - len(args) :])

                try:
                    specific_graph = trace_fn(search_fn_new, sym_args + args)
                except RuntimeError as e:
                    return False

                # correct argnames in the graph
                sym_arg_names = []
                for i, placeholder in zip(
                    range(len(sym_args) + len(args)),
                    specific_graph.graph.nodes,
                ):
                    if i < len(sym_args):
                        sym_arg_names.append(placeholder.target)
                        continue

                    with specific_graph.graph.inserting_after(placeholder):
                        new_node = specific_graph.graph.placeholder(argnames[i - len(sym_args)])
                        new_node.target = new_node.name
                        placeholder.replace_all_uses_with(new_node)
                        specific_graph.graph.erase_node(placeholder)

                argnames = sym_arg_names + argnames
            else:
                try:
                    specific_graph = trace_fn(search_fn, args)
                except RuntimeError as e:
                    return False

            specific_pattern = fx_to_pattern(
                specific_graph,
                argnames=argnames,
                exclusive_arg_names=exclusive_arg_names,
                scalar_workaround=scalar_workaround,
            )

            node = match.output_nodes()[0]
            assert node is not None
            specific_pattern_match = specific_pattern.match(node)

            if is_match(specific_pattern_match) and extra_check(specific_pattern_match):
                # trace the pattern using the shapes from the user program
                match.replacement_graph = trace_fn(replace_fn, args)
                if len(match.nodes) == 1:
                    for n in match.replacement_graph.graph.nodes:
                        _transfer_meta(
                            new_meta=n.meta,
                            old_node=match.nodes[0],
                            pass_name="replacement",
                        )
                return True
            return False

    def normalize_args(**kwargs: Any) -> list[Any]:
        args = [kwargs.pop(name) for name in argnames_static]
        for i in range(1, len(kwargs) + 1):
            if f"tangents_{i}" not in kwargs:
                break
            args.append(kwargs.pop(f"tangents_{i}"))
        assert not kwargs, f"leftover kwargs: {kwargs!r}"
        return args

    if trace_fn is joint_fwd_bwd:
        # If inference mode is enabled during compilation, assume that we don't
        # want to match on any training graph patterns
        if torch.is_inference_mode_enabled():
            return False

    # TODO: Revisit the functionalize_rng_ops for lowmem dropout
    with functorch_config.patch(functionalize_rng_ops=False):
        requires_grad: list[bool] = [isinstance(x, torch.Tensor) and x.requires_grad for x in example_inputs]
        pattern, gm = gen_pattern_and_search_gm(
            search_fn,
            example_inputs,
            trace_fn,
            scalar_workaround,
            exclusive_arg_names,
        )

        print(pattern)

        for pattern_matcher_pass in pass_dicts if isinstance(pass_dicts, Sequence) else [pass_dicts]:
            if isinstance(pattern_matcher_pass, PatternMatcherPass):
                if check_and_add_duplicate_pattern(
                    pattern,
                    gm.graph if gm else None,
                    pattern_matcher_pass.seen_patterns,
                    skip_duplicates=skip_duplicates,
                ):
                    return False

        pattern = ReplacementPatternEntry(
            pattern=pattern,
            extra_check=check_fn,
            normalize_args=normalize_args,
        )
        pattern.register(pass_dicts)
        return pattern.pattern
