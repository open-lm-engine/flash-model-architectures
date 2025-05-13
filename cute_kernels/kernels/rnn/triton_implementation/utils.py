import triton

from ....triton_math import leaky_relu, leaky_relu_backward, sigmoid, sigmoid_backward, tanh, tanh_backward


@triton.jit
def _activation(x, ACTIVATION_FUNCTION, relu_negative_slope):
    if ACTIVATION_FUNCTION == "leaky_relu":
        x = leaky_relu(x, relu_negative_slope)
    elif ACTIVATION_FUNCTION == "sigmoid":
        x = sigmoid(x)
    elif ACTIVATION_FUNCTION == "tanh":
        x = tanh(x)

    return x


@triton.jit
def _activation_backward(y, grad, ACTIVATION_FUNCTION, relu_negative_slope):
    if ACTIVATION_FUNCTION == "leaky_relu":
        grad *= leaky_relu_backward(y, relu_negative_slope)
    elif ACTIVATION_FUNCTION == "sigmoid":
        grad *= sigmoid_backward(y)
    elif ACTIVATION_FUNCTION == "tanh":
        grad *= tanh_backward(y)

    return grad
