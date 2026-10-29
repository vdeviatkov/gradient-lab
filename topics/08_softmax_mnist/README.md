# Softmax regression on MNIST

[All topics](../../README.md#topics) · [Idea and mathematics](idea.md) · [Setup](../../docs/getting-started.md)

**Implementations:** [C++](cpp/README.md) · [Python](python/README.md) · [NumPy](numpy/README.md) · [PyTorch](pytorch/README.md) · [Triton](triton/README.md) · [Backend status](../../README.md#topics)

This is the first milestone that uses a real dataset. It measures how far a purely linear model
gets on MNIST, and reads the confusion matrix to see where it stops.

## Getting the data

MNIST is not committed — `data/` is ignored by Git. Download it first:

```bash
./scripts/download_mnist.sh
```

That fetches the four IDX files into `data/mnist/` and decompresses them. The C++ loader reads
uncompressed IDX directly, so the script gunzips rather than teaching the loader about gzip. Set
`ML_SCRATCH_MNIST_DIR` to read the files from somewhere else. The experiment exits with these
instructions if the data is missing.

## Dataset and preprocessing

60000 training and 10000 test images, 28x28 grayscale, ten digit classes. Preprocessing is a single
documented step: each pixel is rescaled from its stored `0..255` range into `[0, 1]`, applied
identically to every split. Nothing is centered, whitened, deskewed, or augmented.

The last 6000 images of the official training file are held out for validation, leaving 54000 for
training. The official 10000-image test file is loaded at the start but touched exactly once, after
the hyperparameters are fixed. Validation class counts are printed so the split's balance is
visible rather than assumed.

## Method

1. Six combinations of learning rate and L2 coefficient are each trained for 8 epochs on the
   training split and ranked by validation accuracy.
2. The winning combination is retrained for 30 epochs, and wall-clock training time is recorded.
3. The test split is evaluated once: accuracy, macro precision, recall and F1, per-class figures,
   and the full 10x10 confusion matrix.
4. A majority-class baseline predicts the most frequent training digit for every test image.

Mini-batch size is 64 and the seed is `20260911` throughout.

## Running it

```bash
cmake -S . -B build
cmake --build build
./build/cpp_softmax_mnist

# Python backends (after installing the checkout)
python -m ml_scratch.pure.softmax_mnist.experiment      # about four minutes
python -m ml_scratch.numpy.softmax_mnist.experiment     # needs .[numpy]
python -m ml_scratch.pytorch.softmax_mnist.experiment   # needs .[pytorch]
python -m ml_scratch.triton.softmax_mnist.experiment    # Linux + CUDA, needs .[triton]
```

The executable prints only values measured during the current run. It exits unsuccessfully unless
the model beats the majority baseline, reaches at least 90% test accuracy, holds a macro F1 of at
least 0.90, and lowers its training loss over the run.

## Measured result

Recorded on 2026-09-11 with Apple clang 17.0.0 on macOS 26.3.1 (arm64), CMake build type `Release`,
seed `20260911`. The model has **7850 parameters** (784 weights plus a bias for each of ten
classes). Training took **21.74 s** for 30 epochs, 0.72 s per epoch.

Validation search:

| Learning rate | L2 | Train accuracy | Validation accuracy |
|---:|---:|---:|---:|
| 0.10 | 0 | 0.9206 | 0.9335 |
| 0.10 | 1e-4 | 0.9202 | 0.9332 |
| 0.50 | 0 | 0.9235 | **0.9348** |
| 0.50 | 1e-4 | 0.9215 | 0.9338 |
| 0.50 | 1e-3 | 0.9092 | 0.9258 |
| 1.00 | 1e-4 | 0.9028 | 0.9175 |

Regularization did not help here, which is consistent with a 7850-parameter model fitted on 54000
images: it is not in a regime where it can overfit much, and train accuracy (0.9288) ends up close
to test accuracy (0.9221) rather than far above it. The largest learning rate hurt.

Held-out results for the selected configuration (learning rate 0.50, no L2):

| | Test accuracy |
|---|---:|
| majority-class baseline (always 1) | 0.1135 |
| softmax regression | **0.9221** |

Macro precision 0.9223, macro recall 0.9205, macro F1 0.9209. Per-digit F1 ranged from 0.8769 for
the digit 8 to 0.9738 for the digit 1.

| Digit | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0 | 0.9560 | 0.9765 | 0.9662 |
| 1 | 0.9662 | 0.9815 | 0.9738 |
| 2 | 0.8748 | 0.9409 | 0.9066 |
| 3 | 0.8754 | 0.9178 | 0.8961 |
| 4 | 0.9455 | 0.9185 | 0.9318 |
| 5 | 0.9073 | 0.8554 | 0.8806 |
| 6 | 0.9457 | 0.9457 | 0.9457 |
| 7 | 0.9361 | 0.9125 | 0.9241 |
| 8 | 0.9200 | 0.8378 | 0.8769 |
| 9 | 0.8957 | 0.9187 | 0.9070 |

The errors are structured, not random. The most frequent single confusion was **5 predicted as 3**,
53 times, followed by 4 as 9 (41), 7 as 9 (40), and 8 as 3 (40). Those are exactly the digit pairs
whose ink overlaps most in raw pixel space, which is all a per-pixel linear vote can see. Digits 1
and 0, whose ink patterns are most distinct, scored highest.

Two further observations from the run itself:

- Training loss fell from 0.3315 at epoch 1 to 0.2505 at epoch 20, then rose slightly to 0.2532 by
  epoch 30. Mini-batch SGD at a constant learning rate of 0.5 oscillates around the optimum instead
  of settling into it. Nothing here corrects for that — learning-rate schedules and adaptive
  optimizers are later milestones, and this run is the baseline they should be measured against.
- The gap between train (0.9288) and test (0.9221) accuracy is under a point, so the limit is the
  model's expressiveness, not overfitting. A linear model assigns one template per digit and has no
  way to represent relationships between pixels or any invariance to position and stroke width.
  That ceiling is the argument for the MLP and CNN milestones that follow.

## Python, NumPy, PyTorch, and Triton reproduction

Measured on 2026-10-28, seed `20260911`, with the experiment commands listed under
[Running it](#running-it):
- **Pure Python, NumPy, and PyTorch:** CPython 3.14.7, NumPy 2.5.4, and PyTorch 2.14.1 (CPU
  float64) on macOS 26.3.1 (arm64).
- **Triton:** an NVIDIA GeForce RTX 5080 (driver 615.71.09) under Ubuntu 24.04 with CPython
  3.12.3, PyTorch 2.11.0+cu128, and Triton 3.6.0, in float64.

All four runs met every criterion listed above. They printed **identical output** apart from
wall-clock time: the same search table, every per-epoch loss, every metric, and the full confusion
matrix. They share the pixel scaling (`byte / 255.0`), zero initialization, and the shuffle stream.
The shuffle stream is a local `random.Random(20260911)`, not C++'s `std::shuffle`, whose
permutation is not specified across standard libraries, so the trajectory differs from the C++
run above.

| Learning rate | L2 | Train accuracy | Validation accuracy |
|---:|---:|---:|---:|
| 0.10 | 0 | 0.9217 | **0.9340** |
| 0.10 | 1e-4 | 0.9213 | 0.9337 |
| 0.50 | 0 | 0.9255 | 0.9302 |
| 0.50 | 1e-4 | 0.9242 | 0.9288 |
| 0.50 | 1e-3 | 0.9145 | 0.9250 |
| 1.00 | 1e-4 | 0.9104 | 0.9198 |

Validation selected learning rate **0.10**, where C++ selected 0.50. The C++ candidates for 0.10
and 0.50 without L2 were 0.13 points apart, and here they are 0.38 points apart in the other
direction. On a 6000-image validation split, that is 8 and 23 images. The ranking is within the
noise of the shuffle order; the robust conclusions are the same in both runs: L2 did not help,
and a learning rate of 1.0 hurt.

| | Train | Validation | Test |
|---|---:|---:|---:|
| majority-class baseline (always 1) | — | — | 0.1135 |
| softmax regression, learning rate 0.10 | 0.9291 | 0.9373 | **0.9231** |

Macro precision, recall, and F1 were 0.9222, 0.9222, and 0.9220; C++ recorded 0.9223, 0.9205,
and 0.9209. Per-digit F1 ranged from 0.8805 for 8 to 0.9724 for 1, the same extremes as C++.
The most frequent confusion was **9 predicted as 4** (40 times), followed by 2 as 8 (38),
5 as 3 (35), 9 as 7 (34), and 8 as 5 (30). C++'s top four were 5 as 3, 4 as 9, 7 as 9, and 8 as 3.
The same overlapping-ink families dominate both runs, 3/5/8, 4/9, and 7/9, but their order is
not stable across trajectories. Training loss fell from 0.3648 at epoch 1 to 0.2568 at
epoch 30, with small upticks at epochs 15, 23, 27, and 29: even at learning rate 0.10,
constant-step SGD does not settle monotonically. Each rise is 0.0004–0.0012, and the loss still
ends at its lowest point. The C++ run at 0.50 instead rose a net 0.0027 from epoch 20 to 30.

Wall-clock training time for the 30-epoch run was 174.0 s for pure Python, 2.67 s for NumPy,
6.15 s for PyTorch on the CPU, and 9.55 s for Triton on the GPU. These are recorded as measured,
not as a benchmark. The backends ran on different machines; the pure-Python model exploits
sparsity; and the Triton model launches two small float64 kernels per 64-image batch on a
consumer GPU whose float64 rate is a fraction of its float32 rate. See each backend's README for
how it works.
