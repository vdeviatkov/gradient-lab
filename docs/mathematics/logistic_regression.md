# Binary logistic regression mathematics

Binary logistic regression maps an affine score to a probability:

$$
z = w^T x + b, \qquad p(y=1\mid x)=\sigma(z)=\frac{1}{1+e^{-z}}.
$$

The predicted label is one when $p \ge t$, where $t$ is a separately chosen decision threshold.
Changing $t$ changes decisions and classification metrics, but does not retrain the probability
model.

## Loss and explicit gradients

For positive-class weight $\alpha>0$, the implementation minimizes weighted binary cross-entropy:

$$
L = -\frac{1}{N}\sum_i
\left[\alpha y_i\log p_i + (1-y_i)\log(1-p_i)\right].
$$

The default $\alpha=1$ is ordinary binary cross-entropy. The derivative with respect to the score
is $p-y$ in that unweighted case. With weighting it is

$$
\frac{\partial L_i}{\partial z_i}=
\begin{cases}
\alpha(p_i-1) & y_i=1,\\
p_i & y_i=0.
\end{cases}
$$

Therefore, for a mini-batch $B$,

$$
\frac{\partial L}{\partial w_j}
=\frac{1}{|B|}\sum_{i\in B}\frac{\partial L_i}{\partial z_i}x_{ij},
\qquad
\frac{\partial L}{\partial b}
=\frac{1}{|B|}\sum_{i\in B}\frac{\partial L_i}{\partial z_i}.
$$

These loops and updates are written directly in C++. Zero initialization is valid because logistic
regression has no interchangeable hidden units. Seeded shuffling controls mini-batch order.

The sigmoid uses separate formulas for positive and negative scores to avoid overflow. Loss is
computed from logits using the stable identity
$\operatorname{softplus}(z)=\max(z,0)+\log(1+e^{-|z|})$, avoiding `log(0)` at saturated
probabilities.

## Threshold selection and metrics

For binary labels, the confusion matrix counts true negatives (TN), false positives (FP), false
negatives (FN), and true positives (TP). The reported metrics are

$$
\text{accuracy}=\frac{TP+TN}{TP+TN+FP+FN},\quad
\text{precision}=\frac{TP}{TP+FP},\quad
\text{recall}=\frac{TP}{TP+FN},
$$

$$
F_1=2\frac{\text{precision}\cdot\text{recall}}
{\text{precision}+\text{recall}}.
$$

A ratio with a zero denominator is defined as zero by this implementation. This makes the
always-negative classifier's precision, recall, and F1 explicit rather than undefined in output.

Threshold selection evaluates a caller-provided candidate grid on validation data and maximizes
F1. Ties prefer the candidate closest to `0.5`, then the lower candidate. The test set must not be
used for threshold selection. On an imbalanced dataset, a majority classifier can have high
accuracy while recall and F1 reveal that it never detects the minority class.
