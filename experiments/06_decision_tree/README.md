# Experiment 06: decision tree classification

This C++ experiment measures how depth control and pruning change overfitting, rather than
reporting training accuracy alone.

## Dataset

Two classes of 300 samples each are drawn from overlapping Gaussians centered at `(0, 0)` and
`(2.2, 2.2)` with spread `1.2`, described by two informative features followed by three pure noise
features drawn from a standard normal. Each label is then flipped with probability `0.08`.

The flips are part of the data-generating process, so every split carries them at the same rate and
the flipped labels cannot be predicted from the features. That gives the tree something it can only
memorize, never learn — a deep tree will isolate individual flipped points and split on the noise
features, which is exactly the overfitting the experiment measures.

Samples are split by index inside each class into fixed train (150), validation (75), and test (75)
splits, giving 300 / 150 / 150 overall. Seed `20260908` drives `ml_scratch::DeterministicRandom`, a
splitmix64 stream with a Box-Muller transform, so the dataset is identical on every standard
library.

## Method

1. Trees are grown on the training split for `max_depth = 1..12` and without a limit, and train,
   validation, and test accuracy are reported for each.
2. The depth is selected by validation accuracy alone.
3. Gini and entropy are compared at the selected depth.
4. The unlimited tree is post-pruned by reduced-error pruning against the validation split, and
   node count, leaf count, depth, and all three accuracies are reported before and after.
5. A majority-class baseline predicts the most frequent training label for every test sample.

## Running it

```bash
cmake -S . -B build
cmake --build build
./build/cpp_decision_tree
```

The executable prints only values measured during the current run. It exits unsuccessfully unless
the unlimited tree reaches 100% training accuracy, the depth-limited tree generalizes better than
the unlimited one, pruning shrinks the tree without lowering test accuracy, and the selected tree
beats the majority baseline.

## Measured result

Recorded on 2026-09-09 with Apple clang 17.0.0 on macOS 26.3.1 (arm64), CMake build type `Release`,
seed `20260908`. The majority-class baseline reached `0.4867` test accuracy.

| `max_depth` | Leaves | Train | Validation | Test |
|---:|---:|---:|---:|---:|
| 1 | 2 | 0.7600 | 0.7467 | 0.7933 |
| 2 | 4 | 0.8333 | **0.8333** | 0.8600 |
| 3 | 8 | 0.8567 | 0.8133 | 0.8800 |
| 4 | 12 | 0.8567 | 0.8133 | 0.8800 |
| 5 | 18 | 0.8867 | 0.8067 | 0.8733 |
| 6 | 25 | 0.8967 | 0.7200 | 0.7667 |
| 7 | 34 | 0.9333 | 0.7267 | 0.7800 |
| 8 | 40 | 0.9600 | 0.7667 | 0.8200 |
| 9 | 43 | 0.9733 | 0.7400 | 0.8067 |
| 10 | 45 | 0.9867 | 0.7400 | 0.8067 |
| 11 | 47 | 0.9900 | 0.7400 | 0.8067 |
| 12 | 48 | 1.0000 | 0.7400 | 0.8067 |
| unlimited | 48 | 1.0000 | 0.7400 | 0.8067 |

Training accuracy rises monotonically to `1.0000` while validation accuracy peaks at `max_depth = 2`
and then falls by nine points. Reporting the training column alone would rank the unlimited tree
first and the true best model last.

At the selected depth of 2, Gini and entropy produced trees of the same size with close results:
Gini reached `0.8333` validation and `0.8600` test accuracy, entropy `0.7933` and `0.8667`. The gap
is within the noise of a 150-sample split and is not evidence that either criterion is better here.

| Unlimited tree | Nodes | Leaves | Depth | Train | Validation | Test |
|---|---:|---:|---:|---:|---:|---:|
| before pruning | 95 | 48 | 12 | 1.0000 | 0.7400 | 0.8067 |
| after pruning | 13 | 7 | 5 | 0.8567 | 0.8400 | 0.8733 |

Reduced-error pruning collapsed 4 internal nodes, which removed 82 of the 95 nodes because
collapsing a node discards its whole subtree. It traded 14 points of training accuracy for 10
points of validation accuracy and 7 points of test accuracy, and its test accuracy also beat the
best pre-pruned tree in the depth sweep (`0.8733` against `0.8600` at the selected depth). Only the
validation improvement was optimized for; the test figures were measured once, afterwards.
