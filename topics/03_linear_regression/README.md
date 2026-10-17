# Linear regression

[All topics](../../README.md#topics) · [Idea and mathematics](idea.md) · [Setup](../../docs/getting-started.md)

**Implementations:** [C++](cpp/README.md) · [Python](python/README.md) · [Backend status](../../README.md#topics)

Four ways to fit the same line — an exact closed-form reference and three gradient methods — in
pure Python and in C++. The pure-Python experiment then goes after the two numbers that decide
whether gradient descent works at all on a least-squares problem: the largest step it tolerates,
and how many steps it needs. Both are properties of the loss Hessian, both are predicted from it,
and both are then measured.

The fitted relationship is `y = 1.5 + 2x` exactly, over nine training and three test examples
small enough to read, with seed `20260903` for the stochastic methods.

## Running it

Pure Python, no dependencies:

```bash
python -m ml_scratch.pure.linear_regression.experiment
# or, after installing the project:
ml-linear-regression
```

About two seconds. It exits unsuccessfully unless every gradient method reaches the closed-form
coefficients, halving the learning rate reproduces the C++ epoch counts exactly, the measured
stability threshold matches `2 / λ_max`, worse conditioning costs far more epochs, standardizing
gives most of them back, an exactly collinear design is refused, and a near-collinear one blows
the individual coefficients up while their sum stays put.

C++:

```bash
cmake -S . -B build
cmake --build build
./build/cpp_linear_regression
```

The C++ executable runs part one only, and exits unsuccessfully if a gradient-trained model
misses the closed-form coefficients.

## Measured result

Recorded on 2026-10-02 with CPython 3.14.7 and Apple clang 17.0.0 on macOS 26.3.1 (arm64).
Repeat runs reproduce every number exactly.

### Part one: four methods, one answer

| Method | Epochs | Weight | Bias | Train MSE | Test MSE |
|---|---:|---:|---:|---:|---:|
| closed form | — | 2.000000000 | 1.500000000 | 0.0 | 0.0 |
| batch GD | 64 | 2.000000000 | 1.499999058 | 8.9e-13 | 8.9e-13 |
| SGD | 25 | 2.000000039 | 1.499999329 | 4.5e-13 | 4.4e-13 |
| mini-batch GD | 44 | 1.999999990 | 1.499999024 | 9.5e-13 | 9.6e-13 |

Every gradient method lands within **9.8e-07** of the exact coefficients. The closed form is
exact: the minimum of a convex quadratic is where its gradient vanishes, which is a linear
system, so there is nothing to iterate toward.

SGD reaches the target in the fewest *epochs* and the most *updates* — nine per epoch against
batch's one — which is the trade the three methods exist to illustrate and not a claim that SGD
is faster here.

### The two implementations disagreed, and the reason is worth more than the agreement

The C++ milestone reports 135, 51 and 92 epochs for the same three methods, at the same learning
rates, the same seed, the same stopping loss, and from the same zero initialization. For batch
gradient descent there is no randomness at all, so the two should have matched exactly.

They differ because the C++ implementation deliberately minimizes *half* the mean squared error
— a convention [its mathematics documents](idea.md), since
halving does not move the optimum and cancels the factor of two that differentiation produces.
Its update therefore accumulates `residual` where this one accumulates `2 * residual`. The
pure-Python version takes the other convention so that its `gradient` method really is the
gradient of the `mean_squared_error` method beside it, which a central-difference test checks.

Neither is wrong. What they are not is interchangeable: the same nominal learning rate takes a
step twice as large here as there, and quoting the two epoch counts side by side without saying
so would have been a false comparison.

Halving the pure-Python learning rate is exactly what the difference amounts to, and it
reproduces the C++ counts exactly:

| Method | Python at half rate | C++ |
|---|---:|---:|
| batch GD | 135 | 135 |
| SGD | 51 | 51 |
| mini-batch GD | 92 | 92 |

The experiment asserts this match on every run, so if the two implementations ever diverge by
more than this scale factor, it fails rather than quietly reporting incomparable numbers.

### Part two: the largest step that works

The loss is a quadratic, so its Hessian `H = (2/n) X̃ᵀX̃` is one constant matrix. Gradient descent
on a quadratic converges exactly when `0 < learning_rate < 2 / λ_max`, because above that the
error along the steepest direction is multiplied by more than one every step.

Power iteration gives `λ_max = 3.333333333`, so the predicted threshold is **0.600000000**.
Bisecting for the largest rate that still reduces the error over 2000 epochs measures
**0.600001006** — a relative error of **1.7e-06**.

That is the whole of the usual advice about learning rates, on this class of problem, reduced to
a number that can be computed before training starts.

### Part three: what conditioning costs

Multiplying one feature by a constant leaves the problem's meaning untouched and its conditioning
ruined. Training at 90% of each dataset's own stability limit, so the step is always as large as
that dataset allows:

| Feature scale | Condition number | Learning rate | Epochs | Standardized condition | Epochs |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.0 | 5.0e-01 | 56 | 1.1 | 54 |
| 10 | 149.9 | 6.2e-03 | 990 | 1.1 | 56 |
| 100 | 12762.5 | 6.2e-05 | 99481 | 1.1 | 69 |

**The epoch count tracks the condition number almost exactly.** From scale 10 to scale 100 the
condition number rises 85-fold and the epochs rise 100-fold. That is the theory's prediction —
the error contracts by at most `(κ−1)/(κ+1)` per step, so the step count grows with `κ` — arriving
as a measurement.

Standardizing the features — zero mean, unit variance, targets untouched — brings the condition
number to 1.1 in every case and the epoch count from **99481 back to 69**, a factor of 1400. The
gradient's direction is wrong when the coordinates are on different scales, and no choice of a
single shared step size fixes that; rescaling the coordinates does.

### Part four: when the closed form answers a question nobody asked

Two features that are nearly the same carry nearly the same information, and least squares cannot
apportion a coefficient between them. Fitting `y ≈ 1 + 2x₁ + 3x₂` where `x₂ = x₁ ± separation`:

| Separation | Weights | Sum | Train MSE |
|---:|---|---:|---:|
| 1e-02 | (+1.9623, +3.0372) | 4.9995 | 2.019e-07 |
| 1e-04 | (−1.7200, +6.7196) | 4.9995 | 2.019e-07 |
| 1e-06 | (−370.0578, +375.0573) | 4.9995 | 2.019e-07 |
| 1e-08 | refused: singular to working precision | — | — |

Three columns and three different stories. The individual coefficients travel from `(2.0, 3.0)`
to `(−370, +375)` — the second is a hundred times its true value and the first has the wrong
sign. Their **sum stays pinned at 4.9995** across every row, because the sum is what the data
actually determines when the two features move together. And the **fit does not degrade at all**:
the training error is `2.019e-07` in every row, identical to four significant figures.

So nothing here is a numerical failure. The solver returns a correct least-squares solution every
time; the question "what is the coefficient of `x₁`" simply has no answer the data can give, and a
reported number of `−370.0578` is a true statement about one arbitrary member of a flat valley.
At a separation of 1e-08 the valley is flat to working precision and the solver refuses, which is
the one case where the failure is visible without knowing what to look for.

## Limitations

The stability threshold and the conditioning result are exact statements about *quadratic* losses;
nothing here establishes them for the non-quadratic losses of later milestones, where the Hessian
varies from point to point. The conditioning sweep uses one dataset shape and one target
tolerance, so the epochs-versus-κ relationship is measured on three points rather than fitted. The
collinearity demonstration uses one seed and one noise level. And the comparison against C++ is
against one recorded run of that executable, quoted in this file rather than computed here — the
pure-Python half of it is reproducible, the C++ half is a transcription.
