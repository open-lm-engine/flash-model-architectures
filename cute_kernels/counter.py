from typing import Callable


def increment_counter(func: Callable) -> None:
    func._counter = getattr(func, "_counter", 0) + 1
