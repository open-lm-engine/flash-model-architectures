import inspect
import itertools
from collections.abc import Iterable, Sequence
from typing import Any, Callable, Union

import torch
import torch._guards
import torch.fx
from torch._functorch import config as functorch_config
from torch._inductor.pattern_matcher import (
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
    fwd_only,
    fx_to_pattern,
    gen_pattern_and_search_gm,
    is_match,
    joint_fwd_bwd,
)
from torch._prims_common import is_integer_dtype
from torch.fx.experimental.symbolic_shapes import guard_size_oblivious


def register_replacement(
    search_fn: SearchFn,
    replace_fn: ReplaceFn,
    example_inputs: Iterable[Any],
    trace_fn: TraceFn,
    pass_dicts: Union[_PassDictsType, Sequence[_PassDictsType]],
    extra_check: Callable[[Match], bool] = _return_true,
    scalar_workaround: Union[dict[str, Union[float, int]], None] = None,
    exclusive_arg_names: Sequence[str] = (),
    search_fn_pattern: Union[PatternExpr, None] = None,
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
            specific_pattern = search_fn_pattern

            if not specific_pattern:
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
        if search_fn_pattern is None:
            pattern, gm = gen_pattern_and_search_gm(
                search_fn,
                example_inputs,
                trace_fn,
                scalar_workaround,
                exclusive_arg_names,
            )
        else:
            pattern = search_fn_pattern
            gm = None

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
