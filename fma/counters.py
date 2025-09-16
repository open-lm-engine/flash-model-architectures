from collections import defaultdict
from typing import Any


_COUNTERS = defaultdict(int)


def increment_counter(key: Any, value: int = 1) -> None:
    global _COUNTERS
    _COUNTERS[key] += 1


def reset_counters() -> None:
    global _COUNTERS
    _COUNTERS = defaultdict(int)
