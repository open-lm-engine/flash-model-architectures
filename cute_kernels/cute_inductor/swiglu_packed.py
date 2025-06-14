# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import operator

import torch
import torch.nn.functional as F
from torch.fx import Node
from torch.fx.graph_module import GraphModule

from ..ops import swiglu_packed_cute
from .constants import CALL_FUNCTION, CALL_METHOD


def _check_is_valid_chunk(node: Node) -> bool:
    if not (node.op == CALL_METHOD and node.target == torch.chunk.__name__):
        return False

    chunks = node.kwargs.get("chunks", node.args[1])
    if chunks != 2:
        return False

    # dim should be last dim or skip
    if node.kwargs.get("dim", node.args[2]) != -1:
        return False

    return True


def _check_swiglu_after_chunk_and_get_output_node(x: Node, y: Node) -> tuple[bool, Node]:
    # output of chunk should only have 1 user
    if len(x.users) > 1 or len(y.users) > 1:
        return False, None

    silu = list(y.users.keys())[0]
    if not (silu.op == CALL_FUNCTION and silu.target == F.silu):
        return False, None

    # silu output should also have 1 user
    if len(silu.users) > 1:
        return False, None

    # final output
    output = list(silu.users.keys())[0]

    # final output needs multiply
    if not (output.op == CALL_FUNCTION and output.target in [torch.mul, operator.mul]):
        return False, None

    # need to check that the multiply is with x
    if x not in output.all_input_nodes:
        return False, None

    return True, output


def replace_swiglu_unchunked(gm: GraphModule, node: Node) -> None:
    if not _check_is_valid_chunk(node):
        return

    valid, output = _check_swiglu_after_chunk_and_get_output_node(*list(node.users.keys()))
    if not valid:
        return

    print("replacing with swiglu_unchunked_cute")

    with gm.graph.inserting_after(node):
        x = node.kwargs.get("input", node.args[0])
        new_node = gm.graph.call_function(swiglu_unchunked_cute, args=(x,))

    node.replace_all_uses_with(new_node)
    output.replace_all_uses_with(new_node)

    gm.graph.erase_node(node)
    gm.graph.eliminate_dead_code()
