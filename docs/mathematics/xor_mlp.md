# Two-layer MLP mathematics for XOR

A single affine boundary cannot represent XOR. The C++ MLP adds a hidden nonlinear layer with four
units. For input $x = (x_1, x_2)$, the forward pass is

$$
z_j = W^{(1)}_j x + b^{(1)}_j, \qquad
h_j = \tanh(z_j),
$$

$$
z_o = W^{(2)}h + b^{(2)}, \qquad
p = \sigma(z_o) = \frac{1}{1 + e^{-z_o}}.
$$

The probability threshold is $0.5$. Training minimizes mean binary cross-entropy over all four
truth-table examples:

$$
L = -\frac{1}{N}\sum_i \left[y_i \log(p_i) + (1-y_i)\log(1-p_i)\right].
$$

## Explicit gradients

Combining sigmoid with binary cross-entropy simplifies the output derivative:

$$
\frac{\partial L}{\partial z_o} = p-y.
$$

For hidden unit $j$, the chain rule gives

$$
\frac{\partial L}{\partial z_j}
= (p-y)W^{(2)}_j(1-h_j^2).
$$

The weight and bias gradients follow by multiplying each delta by the activation entering that
weight. The implementation accumulates these gradients for all four examples, divides by the
sample count, and performs one full-batch update per epoch.

The convergence rule requires both perfect truth-table accuracy and binary cross-entropy at or
below the configured target. This prevents a lucky threshold crossing from being reported as a
well-trained model. Numerical gradient checking and a reusable layer-oriented backpropagation
design are intentionally reserved for roadmap milestone 3.
