# Experiment 11: initialization, regularization, and normalization

Weight initialization, L1 and L2 penalties, dropout, early stopping, batch normalization, and layer
normalization, each measured against the same unregularized baseline.

## Getting the data

MNIST is not committed. Download it first:

```bash
./scripts/download_mnist.sh
```

## Setup

A `784 -> 128 -> 128 -> 10` network with ReLU hidden layers, trained for 30 epochs at batch size 64
with seed `20260914`. Splits are 2000 training, 2000 validation, and 5000 test images.

The training split is **deliberately small**. With 118282 parameters against 2000 images the
unregularized network memorizes its training data completely, which is the regime where these
techniques are supposed to matter. On the full 54000-image set of
[experiment 09](../09_mlp_mnist/README.md) the train/test gap is under a point and there would be
almost nothing for a regularizer to do.

Every run starts from the same seed, and the `gap` column is train accuracy minus validation
accuracy — the quantity regularization is meant to shrink.

## Running it

```bash
cmake -S . -B build
cmake --build build
./build/cpp_regularization
```

About two and a half minutes for 17 training runs. It exits unsuccessfully unless zero
initialization fails outright, the automatic scheme beats both a too-small and a too-large fixed
scale, the unregularized network actually overfits, at least one regularizer narrows the gap, batch
normalization survives a learning rate that breaks a plain network, and early stopping selects an
epoch before the budget.

## Measured result

Recorded on 2026-09-14 with Apple clang 17.0.0 on macOS 26.3.1 (arm64), CMake build type `Release`,
seed `20260914`. Each run took about 10 seconds. The normalization table was re-recorded on
2026-09-18 after a fix to the training step: until then the optimizer's update was never applied to
the normalization scale and shift, so those five runs had trained with the two frozen at one and
zero. The gradient with respect to them was always computed and checked; it was the final
subtraction that skipped them. Every other table is unaffected, since no other run normalizes.

### Weight initialization

| Scheme | Train loss | Train | Validation | Test | Gap |
|---|---:|---:|---:|---:|---:|
| automatic (He for ReLU) | 0.0195 | 0.9995 | 0.9145 | **0.8726** | 0.0850 |
| Glorot | 0.0365 | 0.9965 | 0.9045 | 0.8624 | 0.0920 |
| fixed uniform 0.001 | 2.2993 | 0.1120 | 0.1045 | 0.1024 | 0.0075 |
| fixed uniform 1.0 | 0.0630 | 0.9830 | 0.7325 | 0.6700 | 0.2505 |
| all zeros | 2.2993 | 0.1120 | 0.1045 | 0.1024 | 0.0075 |

The scale is what matters, and both failure directions are visible. A fixed scale of 0.001 produced
**exactly the same numbers as zero initialization** — train loss 2.2993, just under
$\log 10 = 2.3026$, and predictions collapsed onto a single class. Through two layers a weight of
0.001 shrinks the signal to nothing, so a small-but-nonzero initialization is no better than none at
this depth. A scale of 1.0 does train, reaching 0.9830 train accuracy, but generalizes badly: its
test accuracy is 0.6700 and its gap of 0.2505 is three times the baseline's.

He beat Glorot by about a point, which is the expected direction for ReLU layers but a modest margin
on a network this shallow.

### Regularization

| Penalty | Train loss | Train | Validation | Test | Gap |
|---|---:|---:|---:|---:|---:|
| none | 0.0195 | 0.9995 | 0.9145 | 0.8726 | 0.0850 |
| L2 1e-3 | 0.2883 | 0.9985 | 0.9100 | 0.8686 | 0.0885 |
| L2 1e-2 | 0.8788 | 0.9495 | 0.8780 | 0.8368 | 0.0715 |
| L1 1e-4 | 0.5856 | 0.9985 | 0.9105 | 0.8704 | 0.0880 |
| dropout 0.3 | 0.0393 | 0.9915 | 0.9130 | **0.8774** | 0.0785 |
| dropout 0.5 | 0.0742 | 0.9830 | 0.9120 | 0.8686 | 0.0710 |

This is the section that did not go as the textbook suggests, and it is reported as measured. The
baseline overfits exactly as designed — 0.9995 train against 0.9145 validation — but **no penalty
bought much back**. The best result was dropout at 0.3, and it gained 0.5 points of test accuracy.
L2 at 1e-2 produced the second-narrowest gap (0.0715) by damaging training as much as validation,
and its test accuracy fell by 3.6 points; a narrower gap is not the goal if both ends move down.
Dropout at 0.5 narrowed the gap furthest (0.0710) and still ended level with L2 1e-3 on test.

The reading is that the gap here comes from having 2000 examples rather than from the weights being
too large, and a penalty on weight magnitude cannot manufacture the missing data. Weight decay is
worth its coefficient search when the model is overparameterized *relative to a reasonable amount of
data*; at 59 parameters per training image it is the wrong lever.

### Normalization

| Scheme | Train loss | Train | Validation | Test | Gap |
|---|---:|---:|---:|---:|---:|
| none, lr 0.10 | 0.0195 | 0.9995 | 0.9145 | 0.8726 | 0.0850 |
| batch norm, lr 0.10 | 0.0043 | 1.0000 | 0.9175 | 0.8790 | 0.0825 |
| layer norm, lr 0.10 | 0.0050 | 1.0000 | 0.9205 | 0.8838 | 0.0795 |
| none, lr 1.00 | 1.6778 | 0.3915 | 0.3730 | 0.3840 | 0.0185 |
| batch norm, lr 1.00 | 0.0030 | 0.9990 | 0.9205 | **0.8888** | 0.0785 |

Both normalizations helped at the baseline rate — 0.6 and 1.1 points of test accuracy, and a
quarter of the training loss — and layer normalization edged batch normalization despite needing
no batch statistics and no running averages.

The clearest result in the whole experiment is the last pair. At a learning rate of 1.00 the plain
network **fell apart**, reaching 0.3840 test accuracy, while the batch-normalized one reached
**0.8888** — the best score anywhere in the experiment. Standardizing each layer's pre-activations
removes the coupling between a layer's input scale and the largest stable step, which is precisely
the claim normalization is supposed to support, and here it is worth ten times the usable learning
rate.

Normalization cost nothing measurable in wall-clock time at this size, despite batch normalization
forcing a batch-at-a-time forward and backward pass instead of a sample-at-a-time one.

### Early stopping

Validation loss reached its minimum at **epoch 16**; with patience 5 the run stopped at epoch 21 out
of a 30-epoch budget and restored the epoch-16 parameters.

| | Train | Validation | Test |
|---|---:|---:|---:|
| restored best epoch (16) | 0.9910 | 0.9160 | 0.8710 |
| full 30-epoch run | 0.9995 | 0.9145 | 0.8726 |

Early stopping did what it is specified to do — it found the validation minimum and restored it,
saving nine epochs — and the restored model was **very slightly worse on test**, 0.8710 against
0.8726. That is a difference of eight images out of 5000, well inside the noise of a 2000-image
validation split.

The honest conclusion is that early stopping is a compute saving here, not an accuracy win.
Validation loss is an estimate of generalization from a finite sample, and selecting the argmin of a
noisy curve does not reliably beat simply training to the end when the curve is as flat as this one.
Its value is clearer when the validation curve turns sharply, as it does on the full dataset in
[experiment 09](../09_mlp_mnist/README.md), where the epoch-15 checkpoint was a real improvement
over epoch 25.
