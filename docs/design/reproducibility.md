# Reproducibility

[Design guides](README.md) · [Results policy](../../results/README.md)

Every experiment specifies its dataset, preprocessing and splits, seed, learning rate, epoch budget,
stopping criterion, and software environment. Validation data chooses thresholds and models; the
frozen choice is evaluated on held-out test data. Reproduce a reported result with the command and
configuration in its topic README.

## Randomness

Python models use local `random.Random` instances, and C++ models use local seeded engines.
Repeated runs within a stated implementation/toolchain are tested for matching predictions,
parameters, and training histories where applicable. Never modify a process-wide random stream as
a side effect of fitting a model.

A matching integer seed does not guarantee matching initialization or shuffle order across Python,
C++, or standard-library implementations. C++ distributions and `std::shuffle` need not map engine
outputs identically across toolchains. Experiments that need identical generated data use the
explicit [deterministic data generator](../../src/cpp/include/ml_scratch/deterministic_random.hpp).

## Equivalent implementations

Match datasets, objectives, initialization, update rules, stopping conditions, and metrics before
comparing implementations. State any intentional difference. For example, C++ linear regression
updates on half the MSE while Python differentiates the full MSE; the corresponding learning rates
must differ by a factor of two. See the
[linear-regression experiment](../../experiments/03_linear_regression/README.md).

When exact initial values matter, share explicit parameter values rather than assuming matching
seeds. Use documented tolerances for floating-point comparisons; a change in evaluation order or
compiler optimization can change rounding without changing the underlying algorithm.

## Measurements

Record build type, compiler/runtime, hardware, timing boundaries, warm-up, and repetitions for
performance reports. Preserve raw repetitions where possible. Deterministic model behavior does
not make wall-clock timings deterministic: report variability and refuse unsupported claims.
The [benchmark topic](../../experiments/19_benchmarks/README.md) documents the current harness and
its limitations.

Downloaded datasets and generated checkpoints remain local. A result is reproducible only when
its inputs and procedure are described; committing a generated model is not a substitute.
