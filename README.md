# ml-from-scratch-cpp-python

Machine learning algorithms implemented from scratch in pure Python, NumPy, PyTorch, JAX, C++,
and Triton, with experiments comparing training methods, accuracy, and performance.

## Purpose

This repository is an educational laboratory for understanding how machine-learning algorithms
work below the framework API. Each milestone starts from the mathematics, adds tests and a
reproducible experiment, and then implements equivalent behavior across six complementary
environments:

- **Pure Python** makes every operation visible and keeps the first principles dependency-free.
- **NumPy** introduces vectorized numerical programming without hiding the algorithm.
- **PyTorch** connects the same ideas to tensors, automatic differentiation, and common tooling.
- **JAX** explores functional transformations, automatic differentiation, vectorization, and JIT
  compilation.
- **Modern C++** explores explicit data structures, memory behavior, and compiled performance.
- **Triton** exposes GPU tiling, memory movement, kernel fusion, and hardware-aware optimization.

The goal is not to make six textually identical programs. It is to keep their datasets,
initialization, learning rules, stopping conditions, and metrics comparable enough to explain
meaningful differences.

## Implementation status

| Milestone | Pure Python | NumPy | PyTorch | JAX | C++ | Triton |
|---|---:|---:|---:|---:|---:|---:|
| Perceptron: AND / OR | ✅ Complete | Planned | Planned | Planned | ✅ Complete | — |
| Perceptron: XOR limitation | ✅ Demonstrated | Planned | Planned | Planned | ✅ Demonstrated | — |
| Two-layer MLP: XOR | Planned | Planned | Planned | Planned | ✅ Complete | — |
| Linear regression | Planned | Planned | Planned | Planned | ✅ Complete | — |
| Binary logistic regression | Planned | Planned | Planned | Planned | Planned | — |
| PCA and k-means | Planned | Planned | Planned | Planned | Planned | — |
| Decision tree classification | Planned | Planned | Planned | Planned | Planned | — |
| Backpropagation and gradient checking | Planned | Planned | Planned | Planned | Planned | — |
| Softmax regression: MNIST | Planned | Planned | Planned | Planned | Planned | Planned |
| MLP: MNIST digit recognition | Planned | Planned | Planned | Planned | Planned | Planned |
| Optimizer comparisons | Planned | Planned | Planned | Planned | Planned | Planned |
| Regularization, initialization, and normalization | Planned | Planned | Planned | Planned | Planned | Planned |
| CNN: MNIST digit recognition | Planned | Planned | Planned | Planned | Planned | Planned |
| Residual network: image classification | Planned | Planned | Planned | Planned | Planned | Planned |
| Conditional GAN: MNIST digit generation | Planned | Planned | Planned | Planned | Planned | Planned |
| RNN: character-level sequence modeling | Planned | Planned | Planned | Planned | Planned | Planned |
| LSTM and GRU: character-level sequence modeling | Planned | Planned | Planned | Planned | Planned | Planned |
| Tiny Transformer: character-level language modeling | Planned | Planned | Planned | Planned | Planned | Planned |
| Snake agent: Q-learning and DQN | Planned | Planned | Planned | Planned | Planned | Planned |
| Implementation benchmarks | Planned | Planned | Planned | Planned | Planned | Planned |

`—` means that a standalone Triton kernel would not add meaningful educational value for that
milestone.

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

The implementation contains the gradients needed for this concrete network. The later
backpropagation milestone will turn that focused derivation into a more general design and verify
its gradients numerically.

### Linear regression

The C++ linear regression milestone implements a reusable multivariate model and four fitting
strategies: the normal-equation solution, batch gradient descent, stochastic gradient descent,
and mini-batch gradient descent. The closed-form solver uses pivoted Gaussian elimination written
with standard-library containers; the iterative solvers use explicitly derived gradients and
zero initialization. Seeded shuffling makes SGD and mini-batch runs reproducible.

The experiment fits a small exact line, evaluates on a fixed held-out set, and checks each
optimizer against the closed-form coefficients. See the
[linear regression mathematics](docs/mathematics/linear_regression.md) and the
[reproducible experiment](experiments/03_linear_regression/README.md).

## Testing and quality checks

```bash
pytest
ruff check .
```

Python tests cover the truth-table datasets, predictions, successful AND/OR training, the expected
XOR failure, input validation, and seed reproducibility. CTest covers the same C++ perceptron
behavior, deterministic model parameters and histories, MLP convergence on every XOR example,
all four linear-regression solvers, optimizer agreement, regression metrics and edge cases, and
the project smoke test.

## C++ build

The C++20 project is standard-library-only. It builds a reusable `ml_scratch_cpp` library, three
experiment executables, and CTest executables without downloading a testing framework.

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure

./build/cpp_perceptron_logic_gates
./build/cpp_xor_mlp
./build/cpp_linear_regression
```

## Planned roadmap

1. Perceptron on AND/OR and XOR failure
2. Two-layer MLP for XOR
3. Linear regression using the closed-form solution, batch gradient descent, SGD, and mini-batches
4. Binary logistic regression with threshold selection, confusion matrices, precision, recall, F1,
   and class-imbalance tests
5. PCA for dimensionality reduction and k-means for unsupervised clustering
6. Decision tree classification using Gini impurity and entropy, with depth limits and pruning
7. Manual backpropagation and numerical gradient checking
8. Softmax regression on MNIST
9. MLP for handwritten-digit recognition on MNIST
10. Optimizer comparisons: batch GD, SGD, mini-batch SGD, Momentum, RMSProp, and Adam
11. Weight initialization, L1/L2 regularization, dropout, early stopping, batch normalization, and
   layer normalization
12. Convolutional neural network for handwritten-digit recognition on MNIST
13. Residual connections in an MLP, followed by a small ResNet image classifier on CIFAR-10
14. Conditional GAN for controllable MNIST digit generation, including investigations of training
    instability and mode collapse
15. Character-level RNN with unrolled backpropagation through time and gradient clipping
16. Character-level LSTM and GRU models for longer-term dependencies
17. Tiny decoder-only Transformer with positional encoding, causal masking, and multi-head
    attention
18. Snake environment with a random baseline, tabular Q-learning, and DQN using experience replay,
    a target network, and a documented exploration schedule
19. Reproducible Python, NumPy, PyTorch, JAX, C++, and Triton benchmarks

The classical-ML milestones should use small, inspectable datasets and make preprocessing part of
the experiment. Linear regression should compare optimization against a closed-form reference.
PCA should report measured explained variance, while k-means should document initialization,
inertia, and multiple seeded restarts. Decision trees should demonstrate how depth control or
pruning changes overfitting rather than reporting training accuracy alone.

The RNN and Transformer milestones should share a small fixed text corpus, vocabulary, data split,
and next-character objective so recurrence and self-attention can be compared fairly. The residual
milestone should first isolate an identity skip connection in a small MLP before introducing
convolutional residual blocks. Sequence reports may add cross-entropy, perplexity, parameter count,
and tokens per second, but only when those values have actually been measured.

The GAN milestone should treat generated digits as an evaluation problem, not merely publish a
sample grid. It should compare against a simple baseline and report measured class consistency,
diversity, and coverage using a separately trained classifier or another documented evaluation
method. Generator and discriminator losses alone are not evidence that useful generation was
learned.

Every milestone should use these requirements where applicable:

- a shared dataset representation and behavior contract across implementations;
- fixed train, validation, and test splits;
- deterministic seeds and versioned experiment configuration;
- numerical-stability and edge-case tests;
- a simple baseline that establishes whether learning improved anything;
- checkpoint save/load tests for models that require meaningful training time;
- cross-implementation agreement tests with documented numerical tolerances;
- parameter-count and environment metadata alongside measured results;
- separate compilation, warm-up, and steady-state benchmark timings;
- CPU correctness checks in normal CI and hardware-gated accelerator tests; and
- evaluation that is separate from training, including exploration-free evaluation for
  reinforcement-learning agents.

A status-table cell becomes complete only when its mathematics are documented, the implementation
has automated tests, a reproducible command runs the experiment, and any published result was
actually measured. Serialization and accelerator benchmarks are required only when they apply to
that milestone.

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
