from collections import defaultdict

import torch


class Counter:
    def __init__(self):
        self._counter = defaultdict(int)

    @torch.compiler.set_stance("force_eager")
    def increment(self, key: int, increment: int = 1) -> dict:
        self._counter[key] += increment

    def __getitem__(self, key: str) -> int:
        return self._counter[key]

    def __setitem__(self, key: str, value: int) -> int:
        self._counter[key] = value


_COUNTERS = Counter()


def get_counters() -> Counter:
    global _COUNTERS
    return _COUNTERS
