# Experiment 12: a convolutional network on MNIST

The fully connected milestones left a specific residual: an MLP has no notion of locality or
translation, so its errors spread thinly across many digit pairs. This experiment measures what
building those two ideas into the architecture buys.

## Getting the data

MNIST is not committed. Download it first:

```bash
./scripts/download_mnist.sh
```

## Setup

Every model is trained with **Adam at 0.001, batch 32, 15 epochs, and seed `20260915`**, on 10000
training, 2000 validation, and 5000 test images. Only the architecture differs, so the comparison
isolates it — the optimizer, schedule, preprocessing, and data are held fixed.

The `shifted` column is the interesting one. It evaluates each **already-trained** model on the same
test images moved two pixels down and to the right, with zeros filling the vacated border. Nothing
is retrained and no model ever saw a shifted digit. MNIST digits are size-normalized and centered, so
this measures how much each architecture depends on that centering.

## Running it

```bash
cmake -S . -B build
cmake --build build
./build/cpp_cnn_mnist
```

About three and a half minutes. It exits unsuccessfully unless the CNN matches or beats the MLP with
at least five times fewer parameters, loses less accuracy than the MLP under the shift, beats its own
single-convolution ablation, and beats the linear baseline.

## Measured result

Recorded on 2026-09-15 with Apple clang 17.0.0 on macOS 26.3.1 (arm64), CMake build type `Release`,
seed `20260915`.

| Model | Parameters | Train | Validation | Test | Shifted | Seconds |
|---|---:|---:|---:|---:|---:|---:|
| linear softmax | 7850 | 0.9409 | 0.9200 | 0.8908 | 0.3668 | 1.4 |
| MLP 128 ReLU | 101770 | 0.9991 | 0.9560 | 0.9420 | 0.4306 | 15.5 |
| CNN 8c3-16c3, max pool | **5258** | 0.9960 | 0.9695 | **0.9688** | **0.7126** | 66.7 |

### Parameter efficiency

The CNN reached 0.9688 test accuracy with **5258 parameters** against the MLP's 0.9420 with
**101770** — better accuracy from **19.4x fewer parameters**, and fewer even than the 7850-parameter
linear model. Weight sharing is what does this: a 3x3 kernel is 9 numbers however large the image,
while a dense layer needs one weight per pixel per unit.

### Translation

| Model | Test | Shifted by (2, 2) | Loss |
|---|---:|---:|---:|
| linear softmax | 0.8908 | 0.3668 | 0.5240 |
| MLP 128 ReLU | 0.9420 | 0.4306 | 0.5114 |
| CNN 8c3-16c3 | 0.9688 | 0.7126 | **0.2562** |

Two pixels — under a tenth of the image — cost the MLP more than half its accuracy, and it ends up
barely above the linear model. Both fully connected models learn *where* ink belongs, so moving the
ink invalidates what they learned. The CNN loses half as much. It is not invariant, and the
experiment does not claim it is: a 0.2562 drop is still large, because two 2x2 pooling steps only
absorb displacement up to about four pixels in the deepest layer and the digit's absolute position
still reaches the final dense layer. The honest statement is that convolution plus pooling made the
model markedly less dependent on centering, not independent of it.

### Ablations

| Variant | Parameters | Test | Shifted | Seconds |
|---|---:|---:|---:|---:|
| max pooling (the baseline above) | 5258 | 0.9688 | **0.7126** | 66.7 |
| average pooling | 5258 | 0.9684 | 0.6696 | 66.9 |
| no pooling, stride 2 | 7018 | 0.9512 | 0.5690 | 20.0 |
| one 8-filter layer | 13610 | 0.9510 | 0.5576 | 21.4 |

Max and average pooling were **indistinguishable on centered test data** — 0.9688 against 0.9684,
two images out of 5000 — but differed by 4.3 points once the digits moved. Keeping the strongest
response in a window and discarding its exact location tolerates displacement better than averaging
over it. A comparison on clean test accuracy alone would have concluded the choice does not matter.

Replacing pooling with a stride-2 convolution lost 1.8 points and considerably more robustness,
while using *more* parameters and running three times faster: striding subsamples at a fixed offset
rather than taking the best response in each neighbourhood.

The single-convolution ablation is the counterintuitive one. It has **13610 parameters — more than
twice the two-layer network's 5258** — and scores 1.8 points worse. Adding the second convolution
*reduced* the parameter count, because its pooling step shrinks the feature map before the flatten,
and the final dense layer is where most parameters live.

### Cost

The CNN took 66.7 s against the MLP's 15.5 s: **4.3x the training time for 19.4x fewer parameters**.
This is not a contradiction but the defining trade-off. Weight sharing divides the parameter count by
the number of positions a kernel visits and leaves the arithmetic untouched — every weight is still
used at every position. Parameter count measures memory and statistical capacity; it is not a proxy
for computation, and for convolution the two point in opposite directions.
