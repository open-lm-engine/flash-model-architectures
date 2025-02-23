# Linear layer
Note that the weight matrix $W \in R^{N \times M}$ where $N$ is the output dimension and $M$ is the input dimension. The input $X \in R^{B \times M}$. This layout matches the layout of the PyTorch linear layer.

## Forward
$$y_{ij} = \sum_{k=1}^{K} w_{jk} x_{ik} + b_j$$

## Backward
<!-- x -->
$$\frac{\partial L}{\partial x_{ij}} = \sum_{m=1}^M \sum_{n=1}^N \frac{\partial L}{\partial y_{mn}} \frac{\partial y_{mn}}{\partial x_{ij}}$$
$$= \sum_{m=1}^M \sum_{n=1}^N \frac{\partial L}{\partial y_{mn}} \frac{\partial}{\partial x_{ij}} \left( \sum_{k=1}^K w_{nk} x_{mk} + b_n \right)$$
$$= \sum_{m=1}^M \sum_{n=1}^N \frac{\partial L}{\partial y_{mn}} \sum_{k=1}^K w_{nk} \frac{\partial x_{mk}}{\partial x_{ij}}$$
$$= \sum_{m=1}^M \sum_{n=1}^N \frac{\partial L}{\partial y_{mn}} w_{nj} \frac{\partial x_{mj}}{\partial x_{ij}}$$
$$= \sum_{n=1}^N \frac{\partial L}{\partial y_{in}} w_{nj}$$

<!-- w -->
$$\frac{\partial L}{\partial w_{ij}} = \sum_{m=1}^M \sum_{n=1}^N \frac{\partial L}{\partial y_{mn}} \frac{\partial y_{mn}}{\partial w_{ij}}$$
$$= \sum_{m=1}^M \sum_{n=1}^N \frac{\partial L}{\partial y_{mn}} \frac{\partial}{\partial w_{ij}} \left( \sum_{k=1}^K w_{nk} x_{mk} + b_n \right)$$
$$= \sum_{m=1}^M \sum_{n=1}^N \frac{\partial L}{\partial y_{mn}} \sum_{k=1}^K \frac{\partial w_{nk}}{\partial w_{ij}} x_{mk}$$
$$= \sum_{m=1}^M \sum_{n=1}^N \frac{\partial L}{\partial y_{mn}} \frac{\partial w_{nj}}{\partial w_{ij}} x_{mj}$$
$$= \sum_{m=1}^M \frac{\partial L}{\partial y_{mi}} x_{mj}$$

<!-- b -->
$$\frac{\partial L}{\partial b_i} = \sum_{m=1}^M \sum_{n=1}^N \frac{\partial L}{\partial y_{mn}} \frac{\partial y_{mn}}{\partial b_i}$$
$$= \sum_{m=1}^M \sum_{n=1}^N \frac{\partial L}{\partial y_{mn}} \frac{\partial}{\partial b_i} \left( \sum_{k=1}^K w_{nk} x_{mk} + b_n \right)$$
$$= \sum_{m=1}^M \sum_{n=1}^N \frac{\partial L}{\partial y_{mn}} \frac{\partial b_n}{\partial b_i}$$
$$= \sum_{m=1}^M \frac{\partial L}{\partial y_{mi}}$$

Finally, we have

$$\frac{\partial L}{\partial x_{ij}} = \sum_{n=1}^N \frac{\partial L}{\partial y_{in}} w_{nj}$$
$$\frac{\partial L}{\partial w_{ij}} = \sum_{m=1}^M \frac{\partial L}{\partial y_{mi}} x_{mj}$$
$$\frac{\partial L}{\partial b_i} = \sum_{m=1}^M \frac{\partial L}{\partial y_{mi}}$$
