# Testing and quality checks

[Documentation](README.md) · [Contributing](../CONTRIBUTING.md) · [Test layout](../tests/README.md)

Run commands from the repository root after following [setup](getting-started.md).

## Python and repository tooling

```bash
python -m pytest
ruff check .
python scripts/check_docs.py
```

`pytest` runs topic-local `python/tests/`, shared fixture/numerics tests in `tests/python/`,
and development-tool tests in `tests/tooling/`. Ruff checks Python code. The documentation checker validates local Markdown
file/directory links and Markdown heading anchors in tracked and non-ignored new documentation;
it also rejects maintained documents without a navigation path from the root README.
It does not access the network or validate remote URLs.

Run a focused check while developing:

```bash
python -m pytest topics/04_logistic_regression/python/tests/test_logistic_regression.py
python -m pytest tests/tooling/test_check_docs.py
```

The [shared Python test guide](../tests/python/README.md) describes common fixtures; each topic’s
Python README links its own model and experiment tests.

## C++

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
ctest --test-dir build -R cpp_linear_regression_test --output-on-failure
```

CTest tests use standalone standard-library executables. They need no downloaded datasets and no
GPU. Small IDX/CIFAR/checkpoint fixtures are generated locally by the tests. The
[C++ test guide](../tests/cpp/README.md) describes the coverage and library-boundary checks.

Format changed C++ files with `clang-format -i path/to/file.cpp`; the repository's
[configuration](../.clang-format) is authoritative. Use `git diff --check` before submitting.

## What to verify

- Compare derivatives with independently computed central differences and update rules with
  hand-computed steps. Training loss decreasing alone is not enough.
- Exercise malformed shapes, invalid values, extreme logits, singular systems, and stopping limits.
- Repeat seeded runs and compare predictions, parameters, and histories where applicable.
- Keep equivalent implementations comparable and explain differing loss or learning-rate conventions.
- Check checkpoint round trips, dataset parsing, and model state preservation for relevant changes.
- Keep model selection on validation data and evaluate frozen choices on held-out test data.
- Record algorithm limitations as expected behavior, including the perceptron's XOR failure.

For the C++ suite these principles cover classical models, gradient checks, normalization and
regularization, convolution/residual blocks, adversarial training, recurrent cells, attention,
Snake transitions, Q-learning/replay, and benchmark bookkeeping. See each topic's test source for
its exact assertions.

## Continuous integration

The [CI workflow](../.github/workflows/ci.yml) runs Python tests, Ruff, documentation links, and a
Release CMake build with CTest. Performance measurements belong in reproducible experiments;
wall-clock training or benchmark results are not treated as portable CI expectations.

Repository-layout tests enforce topic-local documentation and package ownership, detect copied
algorithm sources, and verify that benchmark source accounting follows relocated files exactly once.
