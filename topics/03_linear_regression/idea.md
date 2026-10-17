# Linear regression mathematics

[Topic and experiment](README.md) · [C++](cpp/README.md) · [Python](python/README.md)

For a feature vector $x \in \mathbb{R}^d$, linear regression predicts

$$
\hat y = w^T x + b.
$$

The implementation reports mean squared error (MSE),

$$
\operatorname{MSE} = \frac{1}{N}\sum_{i=1}^{N}(\hat y_i-y_i)^2,
$$

and the two implementations descend on it under different but equivalent conventions, which is
worth stating because it changes what a learning rate means.

The C++ implementation minimizes the half-MSE objective $\tfrac12\operatorname{MSE}$. Halving does
not move the optimum and it cancels the factor of two that differentiation produces, so for a
batch $B$ its gradients are

$$
\frac{\partial L}{\partial w_j}
= \frac{1}{|B|}\sum_{i \in B}(\hat y_i-y_i)x_{ij},
\qquad
\frac{\partial L}{\partial b}
= \frac{1}{|B|}\sum_{i \in B}(\hat y_i-y_i).
$$

The pure-Python implementation minimizes the MSE itself, so each of its gradients carries the
factor of two the derivative produces:

$$
\frac{\partial L}{\partial w_j}
= \frac{2}{|B|}\sum_{i \in B}(\hat y_i-y_i)x_{ij},
\qquad
\frac{\partial L}{\partial b}
= \frac{2}{|B|}\sum_{i \in B}(\hat y_i-y_i).
$$

It takes the longer form so that its `gradient` method really is the gradient of the
`mean_squared_error` method beside it, which a central-difference test checks. The consequence is
that the same nominal learning rate takes a step twice as large in Python as in C++, and the
experiment measures exactly that: halving the Python rate reproduces the C++ epoch counts to the
epoch. Neither convention is wrong, and reading the two implementations' epoch counts side by side
without accounting for it would be.

## The Hessian, the largest usable step, and the condition number

Because the objective is quadratic in $\theta$, its second derivative is one constant matrix,

$$
H = \frac{2}{N} \tilde X^T \tilde X,
$$

for the MSE convention, where $\tilde X$ is the design matrix with its leading column of ones.
Two of its properties decide everything about gradient descent on this problem.

**The largest step.** Along an eigenvector of $H$ with eigenvalue $\lambda$, one gradient step
multiplies the error by $1 - \eta\lambda$. Every such factor is below one in magnitude exactly
when $0 < \eta < 2/\lambda_{\max}$, so that is the stability boundary — not a heuristic but an
equality. The pure-Python experiment computes $\lambda_{\max}$ by power iteration, predicts
$2/\lambda_{\max} = 0.600000000$, and measures $0.600001006$ by bisection.

**The number of steps.** With $\kappa = \lambda_{\max}/\lambda_{\min}$, the best fixed step
contracts the error by at most $(\kappa-1)/(\kappa+1)$ per iteration, so the iterations needed for
a fixed accuracy grow with $\kappa$. The experiment multiplies one feature by a constant, which
leaves the problem's meaning untouched and raises $\kappa$ from 2 to 12762, and the epochs rise
from 56 to 99481 — close to the linear growth the bound predicts. Standardizing the features
restores $\kappa \approx 1$ and the epoch count with it.

Full-batch gradient descent uses the entire dataset for one update. Stochastic gradient descent
uses one example per update. Mini-batch gradient descent lies between them. SGD and mini-batch
training shuffle an index vector with a recorded seed; they do not copy or mutate the dataset.
Every gradient-based fit starts from zero weights and bias so optimizer comparisons share the same
initial state.

## Closed-form reference

Appending a column of ones to the design matrix incorporates the bias into a parameter vector
$\theta = (b, w_1, \ldots, w_d)$. Setting the objective's gradient to zero gives the normal
equations

$$
(X^T X)\theta = X^T y.
$$

Both implementations construct these products with loops and solve the system by Gaussian
elimination with partial pivoting, without calling a linear-algebra or machine-learning framework.
Forming $X^T X$ is useful pedagogically but squares the condition number of $X$; QR or SVD is
preferable in production. A rank-deficient design matrix has no unique normal-equation solution,
so both implementations reject it rather than silently selecting one.

Near-rank-deficiency is the harder case, because nothing raises. When two features are nearly
collinear the solution is unique but the valley around it is almost flat, so the individual
coefficients are determined only weakly while the combination the data actually constrains is
determined well. The experiment measures this: as two features are brought from a separation of
$10^{-2}$ to $10^{-6}$, the fitted coefficients travel from $(2.0, 3.0)$ to $(-370, +375)$ while
their sum stays at $4.9995$ and the training error does not change in its first four significant
figures. The lesson is not that the solver failed — it returned a correct least-squares solution
every time — but that "the coefficient of $x_1$" was not a question the data could answer.

## Evaluation

In addition to MSE, the model exposes

$$
R^2 = 1 - \frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2}.
$$

$R^2$ is undefined when every target is identical, and the implementation reports that case as an
input error. Training convergence is explicit: when `target_mse` is positive, fitting stops only
after the measured full-training-set MSE reaches it. A value of zero disables early stopping and
runs the full epoch budget.
