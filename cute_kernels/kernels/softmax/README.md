<!-- **************************************************
Copyright (c) 2025, Mayank Mishra
************************************************** -->

# Softmax

## Forward
$$y_{ij} = \frac{e^{x_{ij}}}{Z_i}$$
where
$$Z_i = \sum_{h=1}^H e^{x_{ih}}$$

## Backward
$$\frac{\partial L}{\partial x_{ij}} = \sum_{b=1}^{B} \sum_{h=1}^{H} \frac{\partial L}{\partial y_{bh}} \frac{\partial y_{bh}}{\partial x_{ij}}$$
$$= \sum_{b=1}^{B} \sum_{h=1}^{H} \frac{\partial L}{\partial y_{bh}} \frac{\partial}{\partial x_{ij}} \left( \frac{e^{x_{bh}}}{Z_b} \right) $$
$$= \sum_{b=1}^{B} \sum_{h=1}^{H} \frac{\partial L}{\partial y_{bh}} \frac{1}{Z_b^2} \left( Z_b \frac{\partial e^{x_{bh}}}{\partial x_{ij}} - e^{x_{bh}} \frac{\partial Z_b}{\partial x_{ij}} \right) $$
$$= \sum_{b=1}^{B} \sum_{h=1}^{H} \frac{\partial L}{\partial y_{bh}} \frac{1}{Z_b} \frac{\partial e^{x_{bh}}}{\partial x_{ij}} - \sum_{b=1}^{B} \sum_{h=1}^{H} \frac{\partial L}{\partial y_{bh}} \frac{e^{x_{bh}}}{Z_b^2} \frac{\partial Z_b}{\partial x_{ij}} $$
$$= \sum_{b=1}^{B} \sum_{h=1}^{H} \frac{\partial L}{\partial y_{bh}} \frac{e^{x_{bh}}}{Z_b} \frac{\partial x_{bh}}{\partial x_{ij}} - \sum_{b=1}^{B} \sum_{h=1}^{H} \frac{\partial L}{\partial y_{bh}} \frac{e^{x_{bh}}}{Z_b^2} \frac{\partial}{\partial x_{ij}} \left( \sum_{k=1}^{H} e^{x_{bk}} \right) $$
$$= \frac{\partial L}{\partial y_{ij}} y_{ij} - \sum_{b=1}^{B} \sum_{h=1}^{H} \frac{\partial L}{\partial y_{bh}} \frac{e^{x_{bh}}}{Z_b^2} \frac{\partial e^{x_{bj}}}{\partial x_{ij}} $$
$$= \frac{\partial L}{\partial y_{ij}} y_{ij} - \sum_{h=1}^{H} \frac{\partial L}{\partial y_{ih}} \frac{e^{x_{ih}}}{Z_i^2} \frac{\partial e^{x_{ij}}}{\partial x_{ij}} $$
$$= \frac{\partial L}{\partial y_{ij}} y_{ij} - \sum_{h=1}^{H} \frac{\partial L}{\partial y_{ih}} \frac{e^{x_{ih}}}{Z_i^2} e^{x_{ij}} $$
$$= \frac{\partial L}{\partial y_{ij}} y_{ij} - \sum_{h=1}^{H} \frac{\partial L}{\partial y_{ih}} y_{ij} y_{ih} $$
$$= y_{ij} \left( \frac{\partial L}{\partial y_{ij}} - \sum_{h=1}^{H} \frac{\partial L}{\partial y_{ih}} y_{ih} \right) $$

Finally, we have

$$\frac{\partial L}{\partial x_{ij}} = y_{ij} \left( \frac{\partial L}{\partial y_{ij}} - \sum_{h=1}^{H} \frac{\partial L}{\partial y_{ih}} y_{ih} \right) $$
