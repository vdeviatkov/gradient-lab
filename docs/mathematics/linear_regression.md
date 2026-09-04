# Linear regression mathematics

For a feature vector $x \in \mathbb{R}^d$, linear regression predicts

$$
\hat y = w^T x + b.
$$

The implementation reports mean squared error (MSE),

$$
\operatorname{MSE} = \frac{1}{N}\sum_{i=1}^{N}(\hat y_i-y_i)^2,
$$

and trains the iterative methods with the equivalent half-MSE objective. The factor of one half
does not change the optimum and cancels the factor of two during differentiation. For a batch
$B$, the explicit gradients are

$$
\frac{\partial L}{\partial w_j}
= \frac{1}{|B|}\sum_{i \in B}(\hat y_i-y_i)x_{ij},
\qquad
\frac{\partial L}{\partial b}
= \frac{1}{|B|}\sum_{i \in B}(\hat y_i-y_i).
$$

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

The C++ implementation constructs these products with loops and solves this linear system using
Gaussian elimination with partial pivoting. It does not call a linear-algebra or machine-learning
framework. Forming $X^T X$ is useful pedagogically but can worsen numerical conditioning; QR or
SVD is preferable in production. A rank-deficient design matrix has no unique normal-equation
solution, so this implementation rejects it rather than silently selecting one.

## Evaluation

In addition to MSE, the model exposes

$$
R^2 = 1 - \frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2}.
$$

$R^2$ is undefined when every target is identical, and the implementation reports that case as an
input error. Training convergence is explicit: when `target_mse` is positive, fitting stops only
after the measured full-training-set MSE reaches it. A value of zero disables early stopping and
runs the full epoch budget.
