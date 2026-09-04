# Experiment 04: binary logistic regression

This C++ experiment uses a small, inspectable one-feature dataset with overlapping classes and a
3:1 negative-to-positive training imbalance. Training, validation, and test examples are fixed in
the source. Mini-batch order uses seed `20260904`.

The model minimizes binary cross-entropy on the training split until it reaches the configured
loss target or epoch limit. It selects a probability threshold from a fixed candidate grid by
maximizing validation F1, freezes that threshold, and evaluates it once on the test split. The
output also includes the conventional `0.5` threshold and an always-negative majority baseline to
show why accuracy alone can mislead on imbalanced data.

Build from the repository root and run:

```bash
cmake -S . -B build
cmake --build build
./build/cpp_logistic_regression
```

The executable prints only values measured during the current run. It exits unsuccessfully unless
training reaches its loss target and the selected classifier improves test F1 over both the
default threshold and the majority baseline.
