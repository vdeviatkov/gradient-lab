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
| Binary logistic regression | Planned | Planned | Planned | Planned | ✅ Complete | — |
| PCA and k-means | Planned | Planned | Planned | Planned | ✅ Complete | — |
| Decision tree classification | Planned | Planned | Planned | Planned | ✅ Complete | — |
| Backpropagation and gradient checking | Planned | Planned | Planned | Planned | ✅ Complete | — |
| Softmax regression: MNIST | Planned | Planned | Planned | Planned | ✅ Complete | Planned |
| MLP: MNIST digit recognition | Planned | Planned | Planned | Planned | ✅ Complete | Planned |
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

The implementation contains the gradients needed for this concrete network. The
[backpropagation milestone](docs/mathematics/backpropagation.md) turns that focused derivation into
a general design and verifies its gradients numerically.

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

### Binary logistic regression

The C++ binary logistic regression model learns probabilities with explicitly implemented
mini-batch gradients and numerically stable binary cross-entropy. It supports a positive-class
weight, seeded shuffling, configurable loss-based stopping, and multivariate inputs. Separate
evaluation utilities calculate a confusion matrix, accuracy, precision, recall, and F1.

The deterministic experiment uses fixed train, validation, and test splits with an imbalanced,
overlapping dataset. It chooses a threshold by validation F1 and compares the frozen threshold on
test data against both `0.5` and an always-negative majority baseline. See the
[logistic regression mathematics](docs/mathematics/logistic_regression.md) and the
[reproducible experiment](experiments/04_logistic_regression/README.md).

### PCA and k-means

The C++ unsupervised milestone implements both algorithms without a linear-algebra dependency. PCA
builds the unbiased sample covariance matrix and diagonalizes it with cyclic Jacobi rotations,
then reports explained variance, explained-variance ratios, and held-out reconstruction error for
each rank. Deterministic conventions — descending eigenvalues with an index tie-break and a fixed
eigenvector sign — make repeated runs comparable.

k-means implements Lloyd's algorithm with both random-sample and k-means++ initialization, seeded
restarts that keep the lowest-inertia run, and explicit empty-cluster handling. Because inertia
falls monotonically with the cluster count, the experiment selects `k` from held-out inertia with
an elbow heuristic and never from labels. Purity against the known groups is computed only at
evaluation time and compared against a single-cluster baseline. See the
[PCA and k-means mathematics](docs/mathematics/pca_kmeans.md) and the
[reproducible experiment](experiments/05_pca_kmeans/README.md).

### Decision tree classification

The C++ decision tree grows axis-aligned splits by exhaustive search over features and midpoint
thresholds, scoring each candidate by impurity decrease under either Gini impurity or entropy.
Sorting with an index tie-break and keeping only strictly better candidates makes the tree a
deterministic function of the dataset. Multiclass labels, `max_depth`, `min_samples_split`,
`min_samples_leaf`, and `min_impurity_decrease` are all supported, and reduced-error post-pruning
collapses subtrees against a validation split.

Growth is greedy, which the tests record as a real limitation rather than a rough edge: on an exact
checkerboard every single-feature split leaves impurity unchanged, so the search stops at the root
even though a depth-two tree would be perfect. The experiment measures the overfitting story
directly — training accuracy climbing to 100% while validation accuracy peaks early and then
falls — instead of reporting training accuracy alone. See the
[decision tree mathematics](docs/mathematics/decision_tree.md) and the
[reproducible experiment](experiments/06_decision_tree/README.md).

### Backpropagation and gradient checking

The C++ backpropagation milestone turns the XOR network's hand-derived gradients into a general
`FeedForwardNetwork`: any stack of fully connected layers, identity, sigmoid, tanh, or ReLU
activations, and squared error, binary cross-entropy, or softmax cross-entropy. The cross-entropy
losses apply their own output transform, so the final layer stays linear, the output delta
collapses to `p - y`, and the loss can be written in the overflow-free softplus and log-sum-exp
forms.

Every parameter is exposed as one flat vector, which is what makes the verification general: any
architecture becomes a function from a parameter vector to a scalar, and the same central-difference
loop checks all of them. The verdict uses the norm ratio rather than a per-element one, and the
experiment measures where the method itself breaks down — the step-size trade-off between
truncation and round-off, and the degeneracy at a converged minimum where every gradient entry
approaches zero. See the
[backpropagation mathematics](docs/mathematics/backpropagation.md) and the
[reproducible experiment](experiments/07_backpropagation/README.md).

### Softmax regression on MNIST

The first milestone on a real dataset. It adds an IDX reader written from scratch — big-endian
headers decoded byte by byte, with explicit rejection of unsupported element types, truncated
payloads, and mismatched image and label counts — and a `SoftmaxRegression` model with stable
log-sum-exp cross-entropy, optional L2 on the weights, and macro-averaged multiclass metrics.

The model computes the same function as a single linear layer of the backpropagation milestone's
network under softmax cross-entropy, and a test requires the two implementations to agree on the
loss, the gradient, and the predicted probabilities. It exists separately because it indexes a
dataset of class indices in place instead of materializing one-hot targets and per-batch copies,
which is what makes the full 60000-image training set practical.

MNIST is not committed; `./scripts/download_mnist.sh` fetches it into the ignored `data/`
directory. The experiment measures where a linear model stops: it reaches 92.21% test accuracy and
its errors are structured — 5 read as 3, 4 as 9 — which is all a per-pixel linear vote can see. See
the [softmax regression mathematics](docs/mathematics/softmax_regression.md) and the
[reproducible experiment](experiments/08_softmax_mnist/README.md).

### An MLP on MNIST

The same `FeedForwardNetwork` from the backpropagation milestone, applied to real data. Three
things were added to make that practical and honest: training overloads that take class indices
instead of one-hot targets and index the dataset in place, so a 54000-image epoch copies nothing
and allocates nothing; text checkpoints that round-trip every parameter exactly at 17 significant
digits; and a confusion matrix on the class-index path. Tests require the class-index and one-hot
paths to produce identical losses, gradients, and training trajectories.

The experiment reuses experiment 08's data, preprocessing, splits, batch size, and seed, and
retrains the linear baseline inside the same run so the comparison is measured rather than quoted.
Validation accuracy is measured after every epoch and the best parameters are checkpointed, then
reloaded from disk to produce the evaluated model. Test accuracy rises from 92.21% to 98.14%, a 76%
reduction in error rate, and the linear model's structured confusions fall furthest: 5 read as 3
drops from 53 cases to 6. See [what a hidden layer adds](docs/mathematics/mlp_mnist.md) and the
[reproducible experiment](experiments/09_mlp_mnist/README.md).

## Testing and quality checks

```bash
pytest
ruff check .
```

Python tests cover the truth-table datasets, predictions, successful AND/OR training, the expected
XOR failure, input validation, and seed reproducibility. CTest covers the same C++ perceptron
behavior, deterministic model parameters and histories, MLP convergence on every XOR example,
all four linear-regression solvers, optimizer agreement, regression metrics and edge cases,
logistic-regression convergence, numerical stability, threshold selection, classification
metrics, class weighting and imbalance behavior, PCA covariance and eigenpair identities,
projection and reconstruction properties, k-means recovery of known clusters, inertia monotonicity,
restart and seed reproducibility, empty-cluster and duplicate-point handling, decision-tree
impurity measures, split selection and tie-breaking, depth and leaf constraints, the greedy
search's checkerboard failure, reduced-error pruning and node compaction, hand-computed neural
network forward passes, losses and gradients, gradient checks across six architecture and loss
combinations, detection of deliberately corrupted gradients, softmax cross-entropy gradients
checked by hand and against central differences, agreement between the softmax model and the
general network, IDX parsing including malformed files, multiclass metrics, agreement between the
class-index and one-hot training paths, checkpoint round trips and rejection of corrupted
checkpoints, and the project smoke test.

## C++ build

The C++20 project is standard-library-only. It builds a reusable `ml_scratch_cpp` library, nine
experiment executables, and CTest executables without downloading a testing framework. The tests
never need a downloaded dataset: the IDX reader is exercised against small files the test writes
itself.

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure

./build/cpp_perceptron_logic_gates
./build/cpp_xor_mlp
./build/cpp_linear_regression
./build/cpp_logistic_regression
./build/cpp_pca_kmeans
./build/cpp_decision_tree
./build/cpp_backpropagation

# Needs MNIST; see scripts/download_mnist.sh
./build/cpp_softmax_mnist
./build/cpp_mlp_mnist
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
