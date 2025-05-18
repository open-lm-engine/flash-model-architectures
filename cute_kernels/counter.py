from typing import Callable


_FUNCTIONS_WITH_COUNTERS = []


def get_counter(func: Callable) -> None:
    return getattr(func, "_counter", 0)


def reset_counter(func: Callable) -> None:
    func._counter = 0


def reset_all_counters() -> None:
    for func in _FUNCTIONS_WITH_COUNTERS:
        func._counter = 0


def increment_counter(func: Callable) -> Callable:
    def _run(*args, **kwargs) -> Callable:
        _run._counter += 1
        return func(*args, **kwargs)

    _run._counter = 0
    _FUNCTIONS_WITH_COUNTERS.append(_run)

    return _run
