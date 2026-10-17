# Linear regression — C++

[Topic and results](../README.md) · [Idea](../idea.md) · [C++ setup and library boundaries](../../../docs/implementations/cpp.md)

## Implementation

Canonical algorithm source:

- [linear_regression.cpp](linear_regression.cpp)

Public interfaces:

- [linear_regression.hpp](include/ml_scratch/linear_regression.hpp)
- [regression.hpp](include/ml_scratch/utils/regression.hpp)

Topic-specific support:

- [regression.cpp](utils/regression.cpp)

The [experiment driver](main.cpp) owns this topic’s data, configuration, evaluation, and reporting.

## Run

From the repository root after the shared C++ build:

```bash
./build/cpp_linear_regression
```

Dataset requirements and measured results are documented in the [topic README](../README.md).

## Tests

- [linear_regression_test.cpp](tests/linear_regression_test.cpp)

These tests are registered with CTest by the local [CMakeLists.txt](CMakeLists.txt).

## Closed-form reference and R²

Optional reference fitting and R² are free utilities in [utils/regression.cpp](utils/regression.cpp),
separate from the gradient-trained model:

```cpp
#include "ml_scratch/linear_regression.hpp"
#include "ml_scratch/utils/regression.hpp"

ml_scratch::LinearRegression model{feature_count};
ml_scratch::fit_closed_form(model, training_data);
double score = ml_scratch::r_squared(model, test_data);
```

The solver writes validated parameters only after a successful solve. MSE remains in the core
model because it is the training objective and stopping metric.
