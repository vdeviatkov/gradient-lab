# Experiment 03: linear regression

This experiment fits the exact relationship `y = 1.5 + 2x` using four from-scratch C++ methods:
a closed-form normal-equation reference, full-batch gradient descent, stochastic gradient descent,
and mini-batch gradient descent. The nine fixed training examples and three held-out test examples
are deliberately small enough to inspect. The stochastic methods use seed `20260903` when
shuffling examples.

Build from the repository root and run:

```bash
cmake -S . -B build
cmake --build build
./build/cpp_linear_regression
```

The executable measures and prints each method's train and test MSE. It exits unsuccessfully if a
gradient-trained model fails to reach the configured MSE or differs from the closed-form
coefficients beyond the documented tolerance. Output numbers are measurements from the current
run; this document intentionally does not embed unexecuted results.
