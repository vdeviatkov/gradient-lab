# Two-layer MLP mathematics for XOR

A single affine boundary cannot represent XOR. The fix is a hidden nonlinear layer, implemented
identically in pure Python and C++ with $h$ units. For input $x = (x_1, x_2)$, the forward pass is

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
well-trained model.

## Why the hidden layer must be wider than one unit

A hidden layer is not enough on its own; it has to be wide enough to produce more than one
boundary. With $h = 1$ the whole model is

$$
p = \sigma\big(w\,\tanh(a \cdot x + b) + c\big),
$$

and $\tanh$ is strictly increasing, as is $\sigma$. Their composition with a scalar multiply is
therefore monotone in the single linear score $a \cdot x + b$, so the set $\{x : p \ge 1/2\}$ is a
half-plane — the same hypothesis class the perceptron had, reached by a longer route. XOR is not
in it. The pure-Python experiment measures the consequence: over fifty seeds a one-unit network
never separates XOR, and most of its failures stop at three of the four rows, the perceptron's
own ceiling.

Two hidden units suffice in principle, since two half-planes can be combined into the XOR region,
but full-batch descent reaches that solution from only some initializations. The failures all
settle at the same place: a loss of $(\ln 2)/2$, with two rows predicted confidently and the other
two left at probability $0.5000$ to four decimals. It is a plateau rather than an exact stationary
point — the measured gradient norm there is about $2.4\times10^{-4}$ — but it is flat enough that
ten thousand epochs do not leave it, and the two undecided rows contribute $\ln 2$ each, which is
where the loss comes from. Four units reach the solution from every seed tried.

## Checking the derivation

Each gradient above is written out by hand, so something independent has to confirm it. Central
differences do:

$$
\frac{\partial L}{\partial \theta_i} \approx \frac{L(\theta + \epsilon e_i) - L(\theta - \epsilon e_i)}{2\epsilon},
$$

which knows nothing about the chain rule and agrees with the analytic gradient to about $10^{-11}$
at $\epsilon = 10^{-5}$. The [backpropagation milestone](backpropagation.md) turns this focused
check into a general facility, including the step-size trade-off that decides how small $\epsilon$
can usefully be.
