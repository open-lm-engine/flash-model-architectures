from enum import Enum


class TritonKernelAlgorithm(Enum):
    full_row_softmax = "full_row_softmax"
    online_softmax = "online_softmax"
