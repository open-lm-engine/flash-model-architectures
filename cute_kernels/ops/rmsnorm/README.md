<!-- **************************************************
Copyright (c) 2025, Mayank Mishra
************************************************** -->

# RMSNorm

## Forward
$$y_{ij} = \frac{w_j x_{ij}}{r_i}$$
where
$$r_i = \sqrt{ \frac{1}{H} \sum_{h=1}^H x_{ih}^2 + \epsilon }$$

## Backward
<!-- weight -->
$$\frac{\partial L}{\partial w_j} = \sum_{b=1}^{B} \sum_{h=1}^{H} \frac{\partial L}{\partial y_{bh}} \frac{\partial y_{bh}}{\partial w_j}$$
$$= \sum_{b=1}^{B} \sum_{h=1}^{H} \frac{\partial L}{\partial y_{bh}} \frac{x_{bh}}{r_b} \frac{\partial w_h}{\partial w_j}$$
$$= \sum_{b=1}^{B} \frac{\partial L}{\partial y_{bj}} \frac{x_{bj}}{r_b}$$

<!-- input -->
$$\frac{\partial L}{\partial x_{ij}} = \sum_{b=1}^{B} \sum_{h=1}^{H} \frac{\partial L}{\partial y_{bh}} \frac{\partial y_{bh}}{\partial x_{ij}}$$
$$= \sum_{b=1}^{B} \sum_{h=1}^{H} \frac{\partial L}{\partial y_{bh}} \frac{w_h}{r_i^2} \left[ r_b \frac{\partial x_{bh}}{\partial x_{ij}} - x_{bh} \frac{1}{2Hr_b} 2 x_{bj} \frac{\partial x_{bj}}{\partial x_{ij}} \right] $$
$$= \frac{\partial L}{\partial y_{ij}} \frac{w_h}{r_i} - \sum_{b=1}^{B} \sum_{h=1}^{H} \frac{\partial L}{\partial y_{bh}} \frac{w_h}{r_i^2} x_{bh} \frac{1}{Hr_b} x_{bj} \frac{\partial x_{bj}}{\partial x_{ij}}$$
$$= \frac{\partial L}{\partial y_{ij}} \frac{w_h}{r_i} - \sum_{h=1}^{H} \frac{\partial L}{\partial y_{ih}} \frac{w_h}{r_i^2} x_{ih} \frac{1}{H} x_{ij}$$
$$= \frac{\partial L}{\partial y_{ij}} \frac{w_h}{r_i} - \frac{1}{Hr_i^3} x_{ij} \sum_{h=1}^{H} \frac{\partial L}{\partial y_{ih}} w_h x_{ih}$$

Finally, we have

$$\frac{\partial L}{\partial w_j} = \sum_{b=1}^{B} \frac{\partial L}{\partial y_{bj}} \frac{x_{bj}}{r_b}$$

$$\frac{\partial L}{\partial x_{ij}} = \frac{\partial L}{\partial y_{ij}} \frac{w_h}{r_i} - \frac{x_{ij}}{Hr_i^3} \sum_{h=1}^{H} \frac{\partial L}{\partial y_{ih}} w_h x_{ih}$$
