from typing import Callable


def increment_counter(func: Callable) -> None:
    counter = getattr(func, "_counter") if hasattr(func, "_counter") else 0
    setattr(func, "_counter", counter + 1)
