# Perceptron mathematics

For an input vector $x$, weights $w$, and bias $b$, the binary perceptron predicts

$$
\hat{y} = \begin{cases}
1 & \text{if } w \cdot x + b \ge 0, \\
0 & \text{otherwise.}
\end{cases}
$$

After a wrong prediction it applies $w \leftarrow w + \eta(y-\hat{y})x$ and
$b \leftarrow b + \eta(y-\hat{y})$, where $\eta > 0$ is the learning rate. If a finite dataset is
linearly separable, repeated updates eventually find a separating boundary.

## Why AND and OR are learnable

For binary inputs, AND can be separated with $x_1 + x_2 - 1.5 \ge 0$, while OR can be separated
with $x_1 + x_2 - 0.5 \ge 0$. These are examples of valid boundaries, not weights hard-coded into
the implementation.

## Why XOR is not learnable by one layer

Suppose a single boundary classified XOR. The positive points `(1, 0)` and `(0, 1)` would require

$$w_1 + b \ge 0 \quad\text{and}\quad w_2 + b \ge 0.$$

Adding these inequalities gives $w_1 + w_2 + 2b \ge 0$. The negative point `(0, 0)` requires
$b < 0$, so this implies $w_1 + w_2 + b > 0`. But the other negative point `(1, 1)` requires
$w_1 + w_2 + b < 0`, a contradiction. A hidden nonlinear layer is needed; that is the next
milestone, not part of the current implementation.
