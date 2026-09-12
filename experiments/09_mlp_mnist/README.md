# Experiment 09: an MLP on MNIST

This experiment puts one hidden layer in front of the softmax classifier of
[experiment 08](../08_softmax_mnist/README.md) and measures what that buys, under identical
conditions.

## Getting the data

MNIST is not committed. Download it first:

```bash
./scripts/download_mnist.sh
```

Set `ML_SCRATCH_MNIST_DIR` to read the IDX files from elsewhere. The experiment exits with these
instructions if the data is missing.

## Method

The dataset, preprocessing (pixels rescaled into `[0, 1]`), splits (54000 / 6000 / 10000), batch
size (64), and seed (`20260911`) are all identical to experiment 08. The softmax baseline is
**retrained inside this experiment** rather than quoted, so both numbers come from the same run on
the same machine.

1. Four architectures are each trained for 5 epochs and ranked by validation accuracy: 64 and 128
   hidden units, ReLU and tanh, learning rates 0.10 and 0.50.
2. The winner is trained for 25 epochs. The epoch loop is driven by the experiment rather than
   inside `fit`, so validation accuracy is measured after every epoch and the parameters are
   checkpointed whenever it improves. Each epoch advances the shuffle seed.
3. The best checkpoint is **reloaded from disk**, and the restored model is the one evaluated. The
   save/load round trip is therefore exercised for real, not only in a unit test.
4. The test split is evaluated once, from the restored checkpoint: accuracy, macro F1, per-digit
   F1, and the full confusion matrix, next to the linear model's.

## Running it

```bash
cmake -S . -B build
cmake --build build
./build/cpp_mlp_mnist
```

Expect roughly four minutes: 25 epochs at about 5.7 s each, plus the architecture search and the
linear baseline. The checkpoint is written to `data/mlp_mnist_best.checkpoint`, inside the ignored
`data/` directory.

The executable prints only values measured during the current run. It exits unsuccessfully unless
the MLP beats the linear model, the restored checkpoint reproduces the best validation accuracy
exactly, and the training loss decreases over the run.

## Measured result

Recorded on 2026-09-11 with Apple clang 17.0.0 on macOS 26.3.1 (arm64), CMake build type `Release`,
seed `20260911`. Training the selected network took **143.57 s** for 25 epochs, 5.74 s per epoch.

Architecture search (5 epochs each):

| Configuration | Parameters | Train loss | Validation accuracy |
|---|---:|---:|---:|
| 64 ReLU, lr 0.10 | 50890 | 0.1081 | 0.9700 |
| 64 tanh, lr 0.10 | 50890 | 0.1330 | 0.9675 |
| 128 ReLU, lr 0.10 | 101770 | 0.0918 | 0.9735 |
| 128 ReLU, lr 0.50 | 101770 | 0.0292 | **0.9792** |

Final run, 128 ReLU at learning rate 0.50:

| Epoch | Training loss | Validation accuracy |
|---:|---:|---:|
| 1 | 0.1445 | 0.9597 |
| 5 | 0.0399 | 0.9732 |
| 10 | 0.0112 | 0.9797 |
| 15 | 0.0025 | **0.9838** |
| 20 | 0.0012 | 0.9833 |
| 25 | 0.0008 | 0.9825 |

Training loss falls to 0.0008 — the network has essentially memorized the training split — while
validation accuracy peaks at epoch 15 and then drifts down. Training loss stops carrying
information about generalization well before the run ends, which is the whole argument for early
stopping. The checkpoint from epoch 15 was restored and reproduced its `0.9838` validation accuracy
exactly.

Held-out comparison, test split evaluated once:

| Model | Parameters | Test accuracy | Macro F1 |
|---|---:|---:|---:|
| softmax regression | 7850 | 0.9221 | 0.9209 |
| MLP, 128 ReLU | 101770 | **0.9814** | 0.9813 |

The test error rate fell from `0.0779` to `0.0186`, a **76.1% reduction**, for 13x the parameters.

Per-digit F1 rose everywhere, and by far the most for the digits the linear model handled worst:

| Digit | Linear | MLP |
|---:|---:|---:|
| 0 | 0.9662 | 0.9863 |
| 1 | 0.9738 | 0.9903 |
| 2 | 0.9066 | 0.9801 |
| 3 | 0.8961 | 0.9802 |
| 4 | 0.9318 | 0.9811 |
| 5 | 0.8806 | 0.9809 |
| 6 | 0.9457 | 0.9843 |
| 7 | 0.9241 | 0.9775 |
| 8 | 0.8769 | 0.9770 |
| 9 | 0.9070 | 0.9753 |

The spread across digits collapsed from `0.8769`–`0.9738` to `0.9753`–`0.9903`.

The specific confusions that limited the linear model fell much further than the overall error rate:

| Confusion | Linear | MLP | Reduction |
|---|---:|---:|---:|
| 5 predicted as 3 | 53 | 6 | 89% |
| 8 predicted as 3 | 40 | 5 | 88% |
| 7 predicted as 9 | 40 | 9 | 78% |
| 4 predicted as 9 | 41 | 11 | 73% |

Three of the four fell by more than the 76% overall reduction. These are the digit pairs whose ink
overlaps most in raw pixel space — precisely what a per-pixel linear vote cannot separate and what
learned hidden features can. The improvement is not a uniform rescaling of a fixed error pattern;
the hidden layer removed the structure that limited the linear model.

What remains is different in character: the MLP's largest single confusion cells are in the single
digits and spread thinly across many pairs, which is the residual a fully connected model still
carries because it has no built-in notion of locality or translation. That is the argument for the
convolutional milestone.
