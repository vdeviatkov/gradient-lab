# Backpropagation and numerical gradient checking

The XOR milestone derived gradients for one fixed network. This milestone generalizes that
derivation to any stack of fully connected layers and then verifies it numerically.

## Forward pass

Write $a^{(0)} = x$ for the input. Layer $l$ holds a weight matrix $W^{(l)}$, a bias vector
$b^{(l)}$, and an elementwise activation $g^{(l)}$:

$$
z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}, \qquad a^{(l)} = g^{(l)}\!\left(z^{(l)}\right).
$$

The implemented activations and their derivatives, written through the activation value where that
is cheaper and better conditioned, are

$$
\begin{aligned}
\text{identity}: &\quad g(z) = z, &g'(z) &= 1,\\
\text{sigmoid}: &\quad g(z) = \sigma(z), &g'(z) &= a(1-a),\\
\text{tanh}: &\quad g(z) = \tanh z, &g'(z) &= 1 - a^2,\\
\text{ReLU}: &\quad g(z) = \max(z, 0), &g'(z) &= \mathbb{1}[z > 0].
\end{aligned}
$$

ReLU has no derivative at $z = 0$; the implementation takes the subgradient $0$ there, which is the
usual convention.

## Backward pass

Let $\delta^{(l)} = \partial L / \partial z^{(l)}$. The chain rule gives the two rules that make up
backpropagation:

$$
\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} \left(a^{(l-1)}\right)^T,
\qquad
\frac{\partial L}{\partial b^{(l)}} = \delta^{(l)},
$$

$$
\delta^{(l-1)} = \left( \left(W^{(l)}\right)^T \delta^{(l)} \right) \odot g'\!\left(z^{(l-1)}\right).
$$

The whole algorithm is one forward sweep that caches every $z^{(l)}$ and $a^{(l)}$, then one
backward sweep applying those three equations. Its cost is the same order as the forward pass,
which is the entire point: the naive alternative of perturbing each parameter separately costs one
forward pass *per parameter*.

## Losses and the output delta

Each loss determines $\delta^{(L)}$ together with the output transform it implies. For $m$ outputs:

**Squared error**, $L = \frac{1}{m}\sum_k (a_k - y_k)^2$, keeps whatever activation the final layer
has, so the chain rule still needs its slope:

$$
\delta^{(L)}_k = \frac{2}{m}\left(a_k - y_k\right) g'\!\left(z^{(L)}_k\right).
$$

**Binary cross-entropy** and **softmax cross-entropy** apply their squashing *inside* the loss, so
the final layer is linear and produces logits. Composing the squashing with the loss makes the
slope factor cancel:

$$
\delta^{(L)}_k = \frac{\sigma(z_k) - y_k}{m},
\qquad
\delta^{(L)}_k = p_k - y_k \ \ \text{with}\ \ p = \operatorname{softmax}(z).
$$

That cancellation is worth stating plainly: $\sigma' = \sigma(1-\sigma)$ vanishes when the unit
saturates, so pairing a sigmoid output with squared error stalls learning on confidently wrong
predictions, while pairing it with cross-entropy does not. Keeping the transform inside the loss
also allows the stable forms

$$
-\log \sigma(z)\ \text{written as}\ \operatorname{softplus}(z) - yz,
\qquad
\log \textstyle\sum_k e^{z_k} = \max_k z_k + \log \sum_k e^{z_k - \max_k z_k},
$$

neither of which overflows or evaluates $\log 0$.

## Numerical gradient checking

Backpropagation is easy to derive and easy to get subtly wrong, so every analytic gradient is
compared against a finite-difference estimate. Taylor expansion gives the two candidates:

$$
\frac{L(\theta + \varepsilon) - L(\theta)}{\varepsilon} = L'(\theta) + O(\varepsilon),
\qquad
\frac{L(\theta + \varepsilon) - L(\theta - \varepsilon)}{2\varepsilon} = L'(\theta) + O(\varepsilon^2).
$$

The symmetric second-order term cancels in the central form, so its truncation error falls by a
factor of 100 per decade of $\varepsilon$ rather than 10. That is why the implementation uses it.

Shrinking $\varepsilon$ does not improve the estimate indefinitely. The numerator subtracts two
nearly equal numbers, and their relative round-off is fixed at machine precision $u$, so the
round-off contribution to the estimate grows like $u/\varepsilon$ while truncation falls like
$\varepsilon^2$. Balancing $\varepsilon^2 \approx u/\varepsilon$ puts the optimum near
$\varepsilon \approx u^{1/3} \approx 6 \times 10^{-6}$ for doubles — which is where the measured
minimum in the experiment lands.

Treating the network as one flat parameter vector is what makes this check general: any
architecture becomes a function $\mathbb{R}^P \to \mathbb{R}$, and the same loop verifies all of
them.

### Comparing the two gradients

The verdict uses the norm ratio

$$
\frac{\lVert a - n \rVert_2}{\lVert a \rVert_2 + \lVert n \rVert_2},
$$

not a per-element ratio. Dividing elementwise by $|a_i| + |n_i|$ makes any parameter whose gradient
happens to be near zero report a large relative error no matter how well the two agree there; the
norm form measures the disagreement against the size of the whole gradient. Both figures are
reported, because the elementwise ones locate a bug once the norm ratio says there is one.

The norm ratio has one degeneracy of its own. At a minimum every entry approaches zero, so there is
nothing left to normalize against and round-off alone pushes the ratio past any tolerance even
though the gradients still agree to several significant figures. Gradient checking therefore
belongs at initialization or mid-training, not at a converged optimum, and the absolute
disagreement is the figure to read there.

One more caveat applies to ReLU. A pre-activation that lands within $\varepsilon$ of the kink has
$L(\theta+\varepsilon)$ and $L(\theta-\varepsilon)$ on opposite sides of a slope discontinuity, so
the difference quotient averages two different slopes and disagrees with either subgradient. That
is a property of the function, not a bug in the gradient. Smooth activations make a clean check.
