# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Any, Dict

import torch
from torch.utils.flop_counter import FlopCounterMode, convert_to_percent_str


aten = torch.ops.aten

io_registry: Dict[Any, Any] = {}


# Define the suffixes for different orders of magnitude
suffixes = ["", "K", "M", "B", "T"]


# Thanks BingChat!
def get_suffix_str(number):
    # Find the index of the appropriate suffix based on the number of digits
    # with some additional overflow.
    # i.e. 1.01B should be displayed as 1001M, not 1.001B
    index = max(0, min(len(suffixes) - 1, (len(str(number)) - 3) // 3))
    return suffixes[index]


def convert_num_with_suffix(number, suffix):
    index = suffixes.index(suffix)
    # Divide the number by 1000^index and format it to two decimal places
    value = f"{number / 1000 ** index:.3f}"
    # Return the value and the suffix as a string
    return value + suffixes[index]


class IOCounterMode(FlopCounterMode):
    def get_flop_counts(self) -> Dict[str, Dict[Any, int]]:
        """Return the flop counts as a dictionary of dictionaries.

        The outer
        dictionary is keyed by module name, and the inner dictionary is keyed by
        operation name.

        Returns:
            Dict[str, Dict[Any, int]]: The flop counts as a dictionary.
        """
        return dict(self.flop_counts)

    def get_table(self, depth=None):
        if depth is None:
            depth = self.depth
        if depth is None:
            depth = 999999

        import tabulate

        tabulate.PRESERVE_WHITESPACE = True
        header = ["Module", "FLOP", "% Total"]
        values = []
        global_flops = self.get_total_flops()
        global_suffix = get_suffix_str(global_flops)
        is_global_subsumed = False

        def process_mod(mod_name, depth):
            nonlocal is_global_subsumed

            total_flops = sum(self.flop_counts[mod_name].values())

            is_global_subsumed |= total_flops >= global_flops

            padding = " " * depth
            values = []
            values.append(
                [
                    padding + mod_name,
                    convert_num_with_suffix(total_flops, global_suffix),
                    convert_to_percent_str(total_flops, global_flops),
                ]
            )
            for k, v in self.flop_counts[mod_name].items():
                values.append(
                    [
                        padding + " - " + str(k),
                        convert_num_with_suffix(v, global_suffix),
                        convert_to_percent_str(v, global_flops),
                    ]
                )
            return values

        for mod in self.flop_counts.keys():
            if mod == "Global":
                continue
            mod_depth = mod.count(".") + 1
            if mod_depth > depth:
                continue

            cur_values = process_mod(mod, mod_depth - 1)
            for value in cur_values:
                values.append(value)

        # We do a bit of messing around here to only output the "Global" value
        # if there are any FLOPs in there that aren't already fully contained by
        # a module.
        if "Global" in self.flop_counts and not is_global_subsumed:
            for idx, value in enumerate(values):
                values[idx][0] = " " + values[idx][0]

            values = process_mod("Global", 0) + values

        if len(values) == 0:
            values = [["Global", "0", "0%"]]

        return tabulate.tabulate(values, headers=header, colalign=("left", "right", "right"))
