<!-- **************************************************
Copyright (c) 2025, Mayank Mishra
************************************************** -->

# SwiGLU

## Forward
$$y_i = u_i g_i \sigma(g_i)$$
where
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

## Backward
<!-- up -->
$$\frac{\partial L}{\partial u_i} = \frac{\partial L}{\partial y_i} g_i \sigma(g_i)$$

<!-- input -->
$$\frac{\partial L}{\partial g_i} = \frac{\partial L}{\partial y_i} u_i \sigma(g_i) \left[1 + g_i - g_i \sigma(g_i) \right]$$
