# Experiment 13: residual connections

An identity skip isolated in an MLP first, then the same idea as convolutional blocks on CIFAR-10.

## Getting the data

Both datasets are downloaded locally and neither is committed:

```bash
./scripts/download_mnist.sh      # part one
./scripts/download_cifar10.sh    # part two, about 170 MB
```

`ML_SCRATCH_MNIST_DIR` and `ML_SCRATCH_CIFAR10_DIR` override the locations. The CIFAR download is
resumable — rerun the script if the transfer is interrupted.

## Part one: isolating the skip

Stacks of 64-unit ReLU layers on 5000 MNIST images, Adam at 0.001, 10 epochs, seed `20260916`.
Three groups differ only in what each middle layer computes:

- **plain**: `g(Wx + b)`
- **residual**: `g(Wx + b + x)`
- **residual + layer normalization**: the same skip, with the pre-activation standardized first

The `initial |grad|` column is the norm of the **first** layer's gradient measured **before any
training**, so it reflects the architecture rather than the fit. It is the quantity depth is supposed
to destroy.

## Part two: convolutional blocks

A stem convolution that halves the resolution, then `n` shape-preserving 3x3 convolutions per stage
with a pooling step between the two stages, on 5000 CIFAR-10 training images at 3x32x32. The plain
and residual networks have **identical shapes and identical parameter counts** — an identity skip
has no parameters of its own — so any difference is attributable to the skip and not to capacity.

The convolutional network carries no normalization, which part one predicts should matter.

## Running it

```bash
cmake -S . -B build
cmake --build build
./build/cpp_resnet_cifar10
```

About seven minutes. It exits unsuccessfully unless the deep plain stack degrades, the skip raises
the early gradient by more than two orders of magnitude, normalization rescues depth where the bare
skip does not, the plain and residual convolutional networks have equal parameter counts, the skip
helps at two blocks per stage, that advantage shrinks at five, and everything beats chance.

## Measured result

Recorded on 2026-09-16 with Apple clang 17.0.0 on macOS 26.3.1 (arm64), CMake build type `Release`,
seed `20260916`. A repeat run reproduced every accuracy, loss, and gradient norm exactly; wall-clock
seconds are the one column that moved, since they depend on what else the machine is doing, and the
figures below come from an otherwise-idle run.

### Part one

**Plain stacks**

| Depth | Parameters | Initial \|grad\| | Train loss | Validation |
|---:|---:|---:|---:|---:|
| 2 | 59210 | 8.065e-01 | 0.0125 | 0.9390 |
| 5 | 71690 | 8.891e-01 | 0.0578 | 0.9340 |
| 10 | 92490 | 3.378e-01 | 0.0488 | 0.9260 |
| 20 | 134090 | 2.862e-01 | 0.1339 | 0.8980 |

**Residual stacks, no normalization**

| Depth | Parameters | Initial \|grad\| | Train loss | Validation |
|---:|---:|---:|---:|---:|
| 2 | 59210 | 2.643e+00 | 0.0103 | 0.9440 |
| 5 | 71690 | 1.310e+01 | 0.0646 | 0.9200 |
| 10 | 92490 | 8.772e+01 | 0.0970 | 0.9200 |
| 20 | 134090 | **4.023e+03** | 0.4439 | **0.7970** |

**Residual stacks with layer normalization**

| Depth | Parameters | Initial \|grad\| | Train loss | Validation |
|---:|---:|---:|---:|---:|
| 2 | 59466 | 2.729e+00 | 0.0078 | 0.9430 |
| 5 | 72330 | 2.442e+00 | 0.0391 | 0.9090 |
| 10 | 93770 | 2.930e+00 | 0.0869 | 0.9120 |
| 20 | 136650 | **4.880e+00** | 0.1990 | **0.9230** |

The plain stack behaves as the degradation story describes: its first-layer gradient falls from
`8.07e-01` to `2.86e-01` and its accuracy drops four points from depth 2 to depth 20, with its
**training** loss rising too — so this is an optimization failure, not overfitting.

**The bare skip does not fix it, and this is the result worth the experiment.** Its first-layer
gradient does not merely survive, it **explodes**: `4.02e+03` at depth 20, four orders of magnitude
above the plain stack's, growing roughly by a factor of ten per five blocks. Its accuracy at depth 20
is `0.7970` — **worse than plain**. The reason is visible in the forward direction as well as the
backward one: each block adds its input back, so activations compound geometrically with depth. An
identity skip on its own trades a vanishing gradient for a growing one.

Adding a normalization inside the block inverts the picture. The first-layer gradient norm stays
essentially **flat from 2 blocks to 20** — `2.73`, `2.44`, `2.93`, `4.88` — because standardizing the
pre-activation discards whatever scale the accumulating identity path has built. It is the only one
of the three whose accuracy does not degrade with depth, and at depth 20 it is the best of the three
by 2.5 points over plain and 12.6 over the bare skip.

The practical reading: a residual block is a skip **and** a normalization, and the pairing is doing
real work rather than being incidental. The original network puts batch normalization in every block
for this reason.

### Part two

| Model | Parameters | Train loss | Train | Validation | Test | Seconds |
|---|---:|---:|---:|---:|---:|---:|
| plain, 2 per stage | 3850 | 1.3702 | 0.5126 | 0.4310 | 0.4430 | 52.8 |
| residual, 2 per stage | 3850 | 1.3106 | 0.5362 | 0.4380 | **0.4665** | 54.1 |
| plain, 5 per stage | 7354 | 1.4381 | 0.4768 | 0.4150 | 0.3840 | 122.8 |
| residual, 5 per stage | 7354 | 1.5224 | 0.4530 | 0.3820 | 0.3650 | 122.7 |

At two blocks per stage the skip is worth **+2.35 points** of test accuracy for exactly the same
3850 parameters. At five blocks it is worth **-1.9 points**, and both networks are worse than their
shallower versions.

That reversal is the prediction part one makes, reproduced in convolutional form: these blocks carry
no normalization either, so the compounding identity path takes over as the blocks accumulate. The
experiment asserts the reversal rather than asserting that residual always wins, which is what makes
it a claim the run could have falsified.

Two honest limitations. The absolute accuracies are low — a 3850-parameter network, 5000 training
images, and 12 epochs is a small fraction of what CIFAR-10 needs, and the comparison between
architectures is what this measures, not the achievable accuracy. And `ConvolutionalNetwork` has no
normalization, so the third variant that rescues depth in part one cannot be run here; adding
per-channel normalization to the convolutional path is the natural next step, and part one is what
isolates the effect it would have.
