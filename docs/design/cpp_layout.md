# C++ core and support boundaries

[Design guides](README.md) · [C++ conventions](../implementations/cpp.md)

Topic-local files are compiled into two aggregate libraries:

| Library | Responsibility |
|---|---|
| `ml_scratch_cpp` | Model state, inference, objectives, explicit gradients, updates, stopping rules |
| `ml_scratch_utils` | Reference solvers, evaluation, diagnostics, checkpoints, readers, reporting |

Support links the core; core training/inference does not link support. The root build owns shared
infrastructure, and each `topics/<topic>/cpp/CMakeLists.txt` adds its own sources and tests.
Public model headers live beside their owner under `cpp/include/ml_scratch/`. The compiler's include
paths preserve `#include "ml_scratch/<module>.hpp"` for users and dependent topics.

## Source ownership

Algorithms stay in their learning topic. Model-specific checkpoint readers, evaluation, and
inspection code sit under that same topic's `cpp/utils/`, so they remain discoverable without
obscuring the algorithm. Mixed-model evaluation and reporting units were split by owner.
Common metrics, dataset readers, fixture generators, data validation, and gradient clipping live
in [shared/cpp](../../shared/cpp/).

The dense network is owned by [backpropagation](../../topics/07_backpropagation/cpp/README.md),
convolution by [CNNs](../../topics/12_cnn_mnist/cpp/README.md), and recurrent cells by
[character RNNs](../../topics/15_char_rnn/cpp/README.md). Later topics use those models directly.
The Transformer workspace is a private declaration shared only with its own diagnostics.

Evaluation needed by training stays in core: XOR convergence checks classification and loss,
decision-tree pruning compares validation accuracy, and Transformer training measures windowed
validation loss. Moving these to utilities would reverse the dependency.

## Optional reference APIs

The [linear-regression reference solver](../../topics/03_linear_regression/cpp/utils/regression.cpp)
and R² are free utilities, rather than model methods:

```cpp
#include "ml_scratch/linear_regression.hpp"
#include "ml_scratch/utils/regression.hpp"

ml_scratch::LinearRegression model{feature_count};
ml_scratch::fit_closed_form(model, training_data);
double score = ml_scratch::r_squared(model, test_data);
```

The solver writes through validated parameter access after a successful solve. MSE remains core
because it is the training objective and stopping metric.

Numerical gradient checks belong to [topic 07's diagnostic utility](../../topics/07_backpropagation/cpp/utils/gradient_check.cpp).
They perturb a copy and leave parameters and normalization state unchanged. Their finite-difference
loop is shared across dense, recurrent, and Transformer checks. Training always uses the explicit
algorithm gradients.

Checkpoint and evaluation member APIs keep their public signatures, with their definitions in
support units. Link `ml_scratch_utils` when calling them. The
[core-only integration test](../../tests/cpp/core_test.cpp) exercises all model families while
linking only `ml_scratch_cpp`, enforcing the dependency boundary.
