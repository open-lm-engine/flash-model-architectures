from collections import defaultdict
from typing import Callable


kernel_counter = defaultdict(int)


def kernel_counter_increment(func: Callable) -> Callable:
    def _call(*args, **kwargs):
        kernel_counter[func.__name__] += 1
        return func(*args, **kwargs)

    return _call
