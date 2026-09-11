# Softmax regression mathematics

Softmax regression — multinomial logistic regression — extends binary logistic regression from two
classes to $K$. It keeps one linear score per class and turns the scores into a distribution.

## Model

For an input $x \in \mathbb{R}^d$, class $k$ has a weight row $w_k$ and a bias $b_k$:

$$
z_k = w_k^T x + b_k, \qquad
p_k = \operatorname{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}} .
$$

The outputs are non-negative and sum to one by construction. With $K = 2$ this reduces to the
sigmoid: $p_1 = 1/(1 + e^{-(z_1 - z_0)})$, so binary logistic regression is the same model written
with the redundant score removed.

That redundancy is worth stating. Softmax is invariant to adding any constant $c$ to every logit,
since $e^{z_k + c}$ scales numerator and denominator alike. The parameterization therefore has one
free direction per input that the data cannot determine, and without a penalty the fitted weights
are unique only up to that shift. An L2 penalty removes the ambiguity by preferring the
smallest-norm representative.

## Loss and gradient

Training minimizes the mean cross-entropy of the true class, with an optional L2 penalty on the
weights:

$$
L = -\frac{1}{N}\sum_{i=1}^{N} \log p_{i, y_i} \; + \; \frac{\lambda}{2}\sum_{k}\lVert w_k \rVert^2 .
$$

Differentiating the composition of softmax and cross-entropy gives the same collapse that binary
logistic regression shows. Writing $y$ for the one-hot label,

$$
\frac{\partial L_i}{\partial z_{ik}} = p_{ik} - y_{ik},
$$

because the $\partial p_j/\partial z_k$ terms telescope against the $1/p_{y}$ factor from the
logarithm. The parameter gradients follow immediately:

$$
\frac{\partial L}{\partial w_k} = \frac{1}{N}\sum_i (p_{ik} - y_{ik})\, x_i + \lambda w_k,
\qquad
\frac{\partial L}{\partial b_k} = \frac{1}{N}\sum_i (p_{ik} - y_{ik}) .
$$

The penalty covers the weights only. A bias sets where a decision boundary sits rather than how
sharply the model responds to an input, so shrinking it does not reduce model complexity — it just
biases the fit toward the origin.

$L$ is convex in the parameters, so there are no spurious local minima and zero initialization is
safe: unlike a hidden layer, the class rows are distinguished by their own labels and have no
symmetry to break.

## Numerical stability

Both $\exp$ and $\log$ are evaluated in shifted form. With $m = \max_j z_j$,

$$
p_k = \frac{e^{z_k - m}}{\sum_j e^{z_j - m}},
\qquad
-\log p_{y} = \Big(m + \log \textstyle\sum_j e^{z_j - m}\Big) - z_{y} .
$$

Both are algebraically identical to the definitions, but every exponent is now at most zero, so
nothing overflows, and the loss is computed from logits without ever evaluating $\log 0$ at a
saturated probability.

## MNIST as a linear problem

Applied to raw MNIST pixels, this model learns one 784-dimensional template per digit plus a bias:
7850 parameters. Each score is the inner product of the image with that template, so the model can
only express a weighted pixel-by-pixel vote. It cannot represent "a closed loop above a vertical
stroke" or any other relationship between pixels, and it has no invariance to translation or
stroke thickness.

That ceiling is the point of the milestone. Around 92% test accuracy is what a linear model reaches
on this data, and the residual errors are structured rather than random: digits whose ink overlaps
heavily in pixel space — 5 against 3, 8 against 3, 4 against 9 — are exactly the pairs a
template-matching model confuses. Reading the confusion matrix rather than the accuracy alone is
what makes the limitation visible, and it is the argument for the hidden layers and convolutions of
the milestones that follow.

## Evaluation

The reported metrics come from a $K \times K$ confusion matrix, `confusion[actual][predicted]`.
Per-class precision and recall are computed from its columns and rows, and the macro averages weight
each class equally, so a rare class counts as much as a common one. A ratio with a zero denominator
is defined as zero, which makes a class the model never predicts explicit in the output instead of
undefined.
