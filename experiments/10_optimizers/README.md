# Experiment 10: optimizer comparisons

Batch gradient descent, SGD, mini-batch SGD, momentum, Nesterov momentum, RMSProp, and Adam, all
minimizing the same loss with the same gradients on the same data from the same starting parameters.

## Getting the data

MNIST is not committed. Download it first:

```bash
./scripts/download_mnist.sh
```

## Setup

A `784 -> 64 ReLU -> 10` network under softmax cross-entropy, on the first 12000 MNIST training
images split into 10000 for training and 2000 for validation. Seed `20260912` fixes both the
initialization and the shuffling, and **every run starts from the identical initial parameters**, so
differences come from the update rule or the batch size and never from where training began.

The subset and the small hidden layer keep the whole comparison — 22 training runs — to about four
minutes. The point here is the relative behavior of the rules, not the best achievable accuracy;
[experiment 09](../09_mlp_mnist/README.md) is the full-data run.

Two choices matter for fairness and are stated rather than assumed:

- **The budget is 20 epochs, not 20 updates.** Equal epochs means equal passes over the data and
  very unequal update counts: full batch gets 20 updates, mini-batches 3140, and batch-1 SGD
  200000. The update count is reported in every row, and so is wall-clock time, so the cost of many
  small updates is visible.
- **Each rule gets a learning rate that suits it.** Adaptive rules normalize the gradient away, so
  their usable rates are an order of magnitude lower. Comparing every rule at one shared rate would
  measure which rule happens to like that rate.

## Running it

```bash
cmake -S . -B build
cmake --build build
./build/cpp_optimizers
```

The executable prints only values measured during the current run. It exits unsuccessfully unless
mini-batches beat full batch by more than 5x at the better of two full-batch rates, batch-1 SGD does
better at the smaller rate than the larger one, momentum improves on plain descent, the adaptive
rules reach above 90% validation accuracy, and Adam's result varies less than plain descent's across
the `0.001`–`0.010` window.

## Measured result

Recorded on 2026-09-12 with Apple clang 17.0.0 on macOS 26.3.1 (arm64), CMake build type `Release`,
seed `20260912`.

### Batch size, plain gradient descent

| Configuration | Updates | Final loss | Best loss | Validation | Seconds |
|---|---:|---:|---:|---:|---:|
| full batch, lr 0.10 | 20 | 1.1465 | 1.1465 | 0.7575 | 10.6 |
| full batch, lr 1.00 | 20 | 1.4844 | 1.2970 | 0.5330 | 10.8 |
| mini-batch 64, lr 0.10 | 3140 | 0.1169 | 0.0745 | 0.9250 | 10.6 |
| stochastic 1, lr 0.10 | 200000 | 0.6380 | 0.5428 | 0.7545 | 13.6 |
| stochastic 1, lr 0.01 | 200000 | **0.0022** | 0.0022 | 0.9600 | 13.6 |

Three things are measured here.

**Full batch is limited by its update count, not its gradient quality.** It computes the exact
gradient every time and still lands 9.8x worse than mini-batches at the same rate. All three
configurations cost about the same wall-clock time, because the work per epoch is dominated by the
forward and backward passes rather than by the parameter updates.

**A larger step does not buy back missing updates.** Raising the full-batch rate from 0.10 to 1.00
made the final loss *worse*, 1.1465 to 1.4844. The stable step size is bounded by the curvature of
the problem, not by how few updates one is willing to take.

**Batch size and learning rate are coupled.** At batch 1 the rate 0.10 reached loss 0.6380 while
0.01 reached 0.0022 — a 290x difference from the rate alone. Batch-1 gradients are far noisier, so
the step that suits a batch of 64 is much too large for a batch of one. Read without the second row,
the first would have wrongly suggested SGD is simply worse than mini-batching.

### Update rule, batch size 64

| Rule | Updates | Final loss | Best loss | Validation | Seconds |
|---|---:|---:|---:|---:|---:|
| gradient descent, lr 0.10 | 3140 | 0.1169 | 0.0745 | 0.9250 | 10.4 |
| momentum, lr 0.10 | 3140 | **0.0010** | 0.0010 | **0.9605** | 10.4 |
| Nesterov momentum, lr 0.10 | 3140 | 0.0010 | 0.0010 | 0.9605 | 10.3 |
| RMSProp, lr 0.001 | 3140 | 0.0310 | 0.0304 | 0.9490 | 10.4 |
| Adam, lr 0.001 | 3140 | 0.0301 | 0.0301 | 0.9475 | 10.3 |

Momentum lowered the training loss by roughly 100x over plain descent at the same rate and the same
number of updates, consistent with the $1/(1-\mu) = 10$ increase in effective step along consistent
directions. Nesterov matched it to four decimal places on this problem; its correction to overshoot
does not bite here.

The adaptive rules reached a *higher* training loss than momentum but comparable validation accuracy,
and all four beat plain descent. The state they carry costs almost nothing in wall-clock terms: the
spread across all five rules is under 2%, because the update is one pass over 50890 parameters
against a forward and backward pass over 10000 samples.

### Sensitivity to the learning rate

Validation accuracy after 20 epochs:

| Rule | lr 0.001 | lr 0.010 | lr 0.100 | lr 1.000 | Spread |
|---|---:|---:|---:|---:|---:|
| gradient descent | 0.8125 | 0.9120 | 0.9250 | 0.8235 | 0.1125 |
| momentum | 0.9105 | 0.9470 | 0.9605 | 0.1075 | 0.8530 |
| Adam | 0.9475 | 0.9460 | 0.7080 | 0.1075 | 0.8400 |

This table contradicts the common claim that adaptive optimizers are less sensitive to the learning
rate, and it is reported as measured rather than reframed. Across the full four-decade sweep, plain
gradient descent has by far the *smallest* spread (0.1125): it degrades gracefully everywhere.
Momentum and Adam both collapse to chance at `lr = 1.000`.

The defensible version of the claim is narrower. Within the decade Adam is actually run in,
`0.001`–`0.010`, its accuracy moves by **0.0015** while plain descent's moves by **0.0995** — about
66x flatter. What Adam offers is insensitivity *inside its working range* together with a good
default, not tolerance of any rate. Its range is shifted down and narrower in absolute terms, because
its step is about `lr` per coordinate whatever the gradient: at `lr = 1` that is a step of size one
in every one of 50890 parameters. Momentum's range is shifted down for the related
$1/(1-\mu)$ reason — running momentum at 0.1 resembles plain descent at 1.0, which is precisely
where plain descent also starts to degrade.

Note also that plain descent's best result here (0.9250 at `lr = 0.100`) is worse than every
adaptive rule's best, so its flat sensitivity curve is flat at a lower level. Robustness and quality
are separate axes, and a single spread number captures only one of them.
