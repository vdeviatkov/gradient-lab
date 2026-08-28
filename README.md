# ml-from-scratch-cpp-python

Machine learning algorithms implemented from scratch in pure Python, NumPy, PyTorch, and C++,
with experiments comparing training methods, accuracy, and performance.

## Purpose

This repository is an educational laboratory for understanding how machine-learning algorithms
work below the framework API. Each milestone starts from the mathematics, adds tests and a
reproducible experiment, and then implements equivalent behavior across four progressively more
specialized environments:

- **Pure Python** makes every operation visible and keeps the first principles dependency-free.
- **NumPy** introduces vectorized numerical programming without hiding the algorithm.
- **PyTorch** connects the same ideas to tensors, automatic differentiation, and common tooling.
- **Modern C++** explores explicit data structures, memory behavior, and compiled performance.

The goal is not to make four textually identical programs. It is to keep their datasets,
initialization, learning rules, stopping conditions, and metrics comparable enough to explain
meaningful differences.

## Implementation status

| Milestone | Pure Python | NumPy | PyTorch | C++ |
|---|---:|---:|---:|---:|
| Perceptron: AND / OR | ✅ Complete | Planned | Planned | ✅ Complete |
| Perceptron: XOR limitation | ✅ Demonstrated | Planned | Planned | ✅ Demonstrated |
| Two-layer MLP: XOR | Planned | Planned | Planned | ✅ Complete |
| Softmax regression: MNIST | Planned | Planned | Planned | Planned |
| MLP: MNIST digit recognition | Planned | Planned | Planned | Planned |
| CNN: MNIST digit recognition | Planned | Planned | Planned | Planned |
| Snake agent with reinforcement learning | Planned | Planned | Planned | Planned |
| Optimizer and implementation benchmarks | Planned | Planned | Planned | Planned |

## Quick start

Python 3.11 or newer is required. The core pure-Python implementation has no runtime
dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

python -m ml_scratch.pure.perceptron_experiment
# Equivalent installed command:
ml-perceptron-logic-gates
```

Optional extras can be installed only when their milestones need them:

```bash
python -m pip install -e ".[numpy]"
python -m pip install -e ".[pytorch]"
python -m pip install -e ".[plotting]"
```

## Current experiments

### A perceptron and logic gates

The first experiment trains a single-layer binary perceptron on the complete truth tables for
AND, OR, and XOR. The implementation uses deterministic seeded initialization and the classic
perceptron update rule

```text
prediction = 1 if w·x + b >= 0 else 0
error      = target - prediction
w          = w + learning_rate * error * x
b          = b + learning_rate * error
```

AND and OR are **linearly separable**: one straight decision boundary can put their positive and
negative examples on opposite sides. The perceptron convergence theorem therefore applies, and
the experiment verifies all four predictions after training.

XOR is **not linearly separable**. Its positive examples `(0, 1)` and `(1, 0)` lie on opposite
corners from its negative examples `(0, 0)` and `(1, 1)`, so no single straight boundary can
separate them. The experiment intentionally stops after a fixed epoch budget and reports this
limitation; it does not claim that XOR was learned. See
[the mathematical explanation](docs/mathematics/perceptron.md) for a short proof.

Run the source-level experiment wrapper after installing the project with:

```bash
python experiments/01_perceptron_logic_gates/run.py
```

The equivalent C++ experiment is built as `cpp_perceptron_logic_gates`. Its dataset, update rule,
stopping condition, deterministic seed, and verification criteria mirror the pure-Python version.

### A two-layer MLP for XOR

The C++ milestone continues from the perceptron's expected XOR failure with a `2 → 4 → 1` network:

- two inputs;
- one hidden layer with four tanh units;
- one sigmoid output;
- binary cross-entropy loss; and
- full-batch gradient descent with explicitly derived gradients.

“Two-layer” counts the two trainable affine transformations (input-to-hidden and
hidden-to-output). The hidden nonlinearities allow the network to form multiple decision
boundaries, so it can represent XOR. Training is considered converged only when all four
predictions are correct and the measured loss reaches the configured threshold. See
[the MLP mathematics](docs/mathematics/xor_mlp.md) for the forward and backward equations.

The implementation contains the gradients needed for this concrete network. Roadmap milestone 3
will turn that focused derivation into a more general backpropagation design and verify its
gradients numerically.

## Testing and quality checks

```bash
pytest
ruff check .
```

Python tests cover the truth-table datasets, predictions, successful AND/OR training, the expected
XOR failure, input validation, and seed reproducibility. CTest covers the same C++ perceptron
behavior, deterministic model parameters and histories, MLP convergence on every XOR example,
invalid configuration, and the project smoke test.

## C++ build

The C++20 project is standard-library-only. It builds a reusable `ml_scratch_cpp` library, two
experiment executables, and CTest executables without downloading a testing framework.

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure

./build/cpp_perceptron_logic_gates
./build/cpp_xor_mlp
```

## Planned roadmap

1. Perceptron on AND/OR and XOR failure
2. Two-layer MLP for XOR
3. Manual backpropagation and numerical gradient checking
4. Softmax regression on MNIST
5. MLP for handwritten-digit recognition on MNIST
6. Train a convolutional neural network for handwritten-digit recognition on MNIST
7. Optimizer comparisons: batch GD, SGD, mini-batch SGD, Momentum, RMSProp, and Adam
8. Snake-playing agent trained with reinforcement learning
9. Python, NumPy, PyTorch, and C++ benchmarks

Comparisons will consider **accuracy, convergence, execution time, memory usage, and
implementation complexity**. Each comparison should document its environment, dataset split,
seed, configuration, and measurement method.

## Results policy

Only results produced by an actually executed, reproducible experiment may be published. Never
invent benchmark numbers, fill a table with estimates, or present an unmeasured claim as a result.
Generated figures and benchmark reports belong under `results/` with enough metadata to reproduce
them; large datasets, trained model artifacts, and build outputs do not belong in Git.

## License

This project is available under the [MIT License](LICENSE).
