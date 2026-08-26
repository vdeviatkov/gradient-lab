# Reproducibility design

Experiments must specify their datasets, seed, learning rate, epoch budget, stopping condition, and
software environment. The current experiment uses an in-process `random.Random` instance so it is
deterministic without modifying Python's global random state. Tests compare training histories,
learned parameters, and predictions from repeated runs with the same seed.

Future cross-language comparisons should define a shared initialization artifact when identical
starting values matter; using the same integer seed alone does not guarantee that different random
number generators produce identical values.
