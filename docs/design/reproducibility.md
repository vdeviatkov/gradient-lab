# Reproducibility design

Experiments must specify their datasets, seed, learning rate, epoch budget, stopping condition, and
software environment. The current experiment uses an in-process `random.Random` instance so it is
deterministic without modifying Python's global random state. Tests compare training histories,
learned parameters, and predictions from repeated runs with the same seed.

Future cross-language comparisons should define a shared initialization artifact when identical
starting values matter; using the same integer seed alone does not guarantee that different random
number generators produce identical values.

The C++ experiments use local `std::mt19937` engines and never modify shared random state. Tests
train independent models with the same seed and require identical histories and parameters. C++
standard-library implementations may map engine outputs differently for distributions and
shuffling, so exact values are reproducible within a toolchain but are not claimed to match Python
or every C++ standard library. A future benchmark suite should serialize shared initial parameters
when cross-language identity is required.
