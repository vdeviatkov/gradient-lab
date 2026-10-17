# C++ core and experiment support

[Design guides](README.md) · [C++ implementation](../../src/cpp/README.md)

`src/cpp/src/` contains the algorithms: model state, forward passes, losses, explicit derivatives,
updates, stopping rules, validation, and shared operations required during training or inference.
`src/cpp/utils/` contains the surrounding experiment tools. CMake builds them as separate static
libraries: `ml_scratch_cpp` and `ml_scratch_utils`. The dependency goes from utilities to core.
Experiment and existing model-test targets link `ml_scratch_utils`, which brings in the core.
A caller using training and inference alone can link just `ml_scratch_cpp`.

The `cpp_core_test` executable deliberately links only the core and exercises every model's
training path, including the GAN and DQN. This catches accidental runtime dependencies back into
support code. `cpp_utilities_test` checks the extracted reference and diagnostic APIs; the existing
metric, checkpoint, loader, and benchmark tests continue to cover the relocated implementations.

## Boundaries found during the source audit

Every original C++ translation unit was checked for responsibilities outside its algorithm.

| Original module | Core responsibility | Support moved to `utils/` |
|---|---|---|
| `linear_regression` | Prediction, MSE objective, gradient training, parameter access | Normal-equation reference solver and R² in `regression.cpp` |
| `logistic_regression` | Logits, probabilities, BCE, gradient training | Binary metrics, threshold tuning, evaluation |
| `softmax_regression` | Scores, probabilities, cross-entropy, gradient training | Multiclass metrics, accuracy, confusion matrix, evaluation |
| `neural_network` | Forward/backward passes, normalization, dropout, regularization, training | Numerical gradient checks, accuracy, confusion matrix, checkpoint I/O |
| `convolution` | Layer construction, convolution/pooling/dense passes, gradients, training | Accuracy, confusion matrix, checkpoint I/O |
| `rnn` | Recurrent cells, BPTT, truncation, training, generation | Numerical derivatives, gradient-reach diagnostics, checkpoint I/O, cell names |
| `transformer` | Attention, normalization, gradients, windowed loss, training, generation | Numerical derivatives, attention inspection, checkpoint I/O |
| `gan` | Conditional sampling, adversarial losses and gradients, alternating training | Checkpoint I/O, loss names |
| `reinforcement` | Exploration, Q updates, replay, target-network updates | Checkpoint I/O; greedy episode evaluation already lives outside the implementation |
| `decision_tree` | Impurity, split search, prediction, pruning | Depth and leaf-count diagnostics |
| `pca` | Covariance, Jacobi eigensolver, fit, projection and reconstruction | Reconstruction error and cumulative variance reporting |
| `kmeans` | Initialization, assignment, centroid updates, restarts, inertia | Label-based cluster purity |
| `perceptron` | Prediction and perceptron updates | Accuracy reporting |
| `xor_mlp` | Two-layer forward/backward passes, loss, training | No separate support was mixed in; accuracy is used by convergence |
| `dataset` | Shared data representations, validation, column means, distance | Class counts and one-hot preprocessing |
| `text_corpus` | N-gram algorithm now in `src/char_ngram.cpp` | Vocabulary, encoding, splits, loading, perplexity and bits-per-character reporting |
| `snake` | State transitions, food placement, observations | ASCII rendering |
| `optimizer` | Update rules and optimizer state | Display names |
| `deterministic_random` | — | Experiment data generation moved intact |
| `logic_gates` | Data types remain shared | Truth-table fixtures and gate names moved intact |
| `mnist`, `cifar10` | — | Dataset readers moved intact |
| `benchmark`, `project_info` | — | Timing, environment/source measurement, JSON and metadata moved intact |

Gradient clipping is used by the RNN, Transformer, and DQN, so it belongs in core
`gradient.cpp` / `gradient.hpp`, independently of any one model. The Transformer workspace layout
is shared privately with its diagnostic implementation through `internal/transformer_workspace.hpp`.
It is not part of the public API.

Training-dependent evaluation stays with its algorithm: XOR requires both perfect classification
and a loss target to converge, decision-tree pruning compares validation accuracy, and Transformer
training measures validation loss using its context-window policy. Moving these into the support
library would reverse the dependency or duplicate the algorithm's behavior.

## API changes

The linear model no longer exposes the optional reference solver and R² as member functions:

```cpp
#include "ml_scratch/linear_regression.hpp"
#include "ml_scratch/utils/regression.hpp"

ml_scratch::LinearRegression model{feature_count};
ml_scratch::fit_closed_form(model, training_data);
double score = ml_scratch::r_squared(model, test_data);
```

The reference solver writes through a validated `set_parameters` method and leaves the model
unchanged if solving fails. MSE remains a core method because it is the training objective and
stopping metric.

Numerical parameter gradients and their comparisons are free utilities in
`ml_scratch/utils/gradient_check.hpp`:

```cpp
auto numerical = ml_scratch::numerical_gradient(network, data);
auto check = ml_scratch::check_gradient(network.gradient(data), numerical);
```

They perturb a copy, so a check leaves model parameters and normalization statistics unchanged.
The same central-difference loop serves dense, recurrent, and Transformer models; every derivative
used for training remains explicitly implemented in core.

Existing checkpoint and evaluation member APIs retain their signatures, with their definitions
compiled only into the utilities library. Include `ml_scratch/utils/classification_metrics.hpp`
when using model-independent binary or multiclass metric functions. Existing dataset-reader,
benchmark, and fixture header paths remain stable.

The experiments retain their datasets, seeds, update equations, and stopping rules. Benchmark
source-size accounting includes the relocated support rather than presenting relocation as an
implementation-size improvement.
