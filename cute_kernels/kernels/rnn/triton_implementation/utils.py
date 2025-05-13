import triton

from ....triton_math import leaky_relu, sigmoid, tanh


@triton.jit
def _activation(x, ACTIVATION_FUNCTION, relu_negative_slope):
    if ACTIVATION_FUNCTION == "leaky_relu":
        x = leaky_relu(x, relu_negative_slope)
    elif ACTIVATION_FUNCTION == "sigmoid":
        x = sigmoid(x)
    elif ACTIVATION_FUNCTION == "tanh":
        x = tanh(x)

    return x
