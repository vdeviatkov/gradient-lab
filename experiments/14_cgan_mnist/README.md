# Experiment 14: a conditional GAN on MNIST

Two networks trained against each other: a generator that turns noise and a requested digit into an
image, and a discriminator that tries to tell its images from real ones. Generated digits are
treated as an evaluation problem — scored by a separately trained classifier and by distance-based
measures of variety and coverage, against two baselines — because the two players' losses say
nothing about whether generation was learned.

## Getting the data

MNIST is not committed. Download it first:

```bash
./scripts/download_mnist.sh
```

`ML_SCRATCH_MNIST_DIR` overrides the location.

## Setup

The first 10000 training images are used both to train the GAN and to train the judge; the next
2000 are held out and never trained on by either. Seed `20260917` throughout.

**The judge** is a `784 → 128 ReLU → 10` `FeedForwardNetwork`, Adam at 0.001, batch 32, 5 epochs.
Its held-out accuracy is reported so its verdicts can be read against how good a judge it is. It
never sees a generated image while training.

**The generator** takes 64 standard-normal noise values with a one-hot label appended, `74 → 256
leaky ReLU → 784 sigmoid`. **The discriminator** takes an image with the same one-hot label
appended, `794 → 256 leaky ReLU → 1 logit`. Both use Adam at 0.001 with `beta1 = 0.5`, batch 64,
and 30 epochs. The generator minimizes the non-saturating loss `-log D(G(z))`.

### Metrics

Every generator is scored on 100 samples per class, drawn from the same noise so that only the
label differs between rows:

- **consistency** — the fraction the judge assigns to the class they were generated for. Chance is
  0.1.
- **diversity** — the mean pairwise distance between samples of one class, as a fraction of the real
  held-out images' figure. Zero means one image per class; one means as varied as the data.
- **covered** — classes for which at least half the samples are judged correctly.
- **coverage** — mean distance from each real held-out image to its nearest sample of the same
  class. Lower means the samples reach more of the data.
- **novelty** — mean distance from each sample to its nearest real *training* image of the same
  class. A generator that memorized the training set scores near zero; the real held-out images'
  own figure is the yardstick for a sample that is new but on the data manifold.

### Baselines

- **class mean** — the training mean image of each class, repeated. The most recognizable single
  image a class has and no variety at all.
- **independent pixels** — each pixel drawn from its class's per-pixel mean and standard deviation,
  clipped to `[0, 1]`. Full per-pixel variety and no model of how pixels go together.

The real held-out images are scored too, as the ceiling.

### Part two: the minimax loss

The same networks from the same initialization, trained for 20 epochs with the generator
minimizing `log(1 - D(G(z)))` instead — the objective as the minimax game states it, whose gradient
is proportional to `D(G(z))` and so fades when the discriminator is winning. The generator's gradient
norm and `D(G(z))` are recorded every epoch for both runs.

## Running it

```bash
cmake -S . -B build
cmake --build build
./build/cpp_cgan_mnist
```

About seven minutes. It writes a sample grid to `results/figures/14_cgan_mnist_samples.png`
(`ML_SCRATCH_FIGURE_PATH` overrides it) and a checkpoint to `data/cgan_mnist.checkpoint`
(`ML_SCRATCH_CGAN_CHECKPOINT`), and prints one sample per class as text. It exits unsuccessfully
unless the judge is at least 90% accurate, the trained generator's consistency is above 0.8 and above
the untrained one's, every class is covered, diversity stays above half the data's, coverage beats
the independent-pixel baseline, samples are at least half as far from their nearest training image
as real held-out images are, the minimax generator's gradient is smaller and its consistency lower
than the non-saturating run's at the same epoch, and the discriminator never scores the fakes below
0.05 on average.

## Measured result

Recorded on 2026-09-17 with Apple clang 17.0.0 on macOS 26.3.1 (arm64), CMake build type `Release`,
seed `20260917`. A repeat run reproduced every loss, score, gradient norm, and metric exactly;
wall-clock seconds are the one column that moves between runs. Total wall time 7 minutes 8 seconds.

The judge scored **0.9515** on the held-out images.

### Baselines and the trained generator

| Model | Consistency | Diversity | Covered | Coverage | Novelty |
|---|---:|---:|---:|---:|---:|
| real held-out images | 0.9540 | 1.0000 | 10 | 0.0000 | 4.7615 |
| class-mean baseline | 1.0000 | 0.0000 | 10 | 6.4071 | 4.7117 |
| independent-pixel baseline | 0.9680 | 0.7768 | 10 | 7.4982 | 6.8418 |
| untrained generator | 0.1000 | 0.6152 | 1 | 13.7272 | 13.7398 |
| **trained generator** | **0.9420** | **0.8787** | **10** | **6.9423** | **6.0428** |

The judge assigns **94.2%** of generated images to the digit they were asked for, from a starting
point of exactly chance — the untrained generator produces the same blob for every label. Every
class is covered. Within-class diversity is **0.88** of the real images': the generator uses its
noise rather than emitting one image per class. Coverage beats the independent-pixel baseline
(6.94 against 7.50), and novelty is **above** the real held-out images' own figure (6.04 against
4.76): generated digits are, if anything, farther from their nearest training image than a
genuinely new digit is, so nothing has been memorized.

![One row per class, eight samples across](../../results/figures/14_cgan_mnist_samples.png)

Two baseline numbers deserve a look before reading the generator's as a win.

The **class mean scores 1.000 on consistency**, better than the real images, and nothing else in
the table comes close. It also has the best coverage, 6.41 against the generator's 6.94. Both
follow from the same fact: the class mean is the image that minimizes expected squared distance to
the class, so any nearest-neighbour or L2 measure rewards it, and the judge finds an average digit
easier to read than most real ones. Its diversity of **0.000** is the whole story — it is one
picture per class. A generator is not trying to beat the mean at being the mean; it is trying to
produce many distinct images that each read as the digit, which is why the metrics only mean
something together. The experiment therefore requires coverage to beat the independent-pixel
baseline and reports the class mean without requiring it.

The **independent-pixel baseline scores 0.968 on consistency** — also above the real images.
Per-pixel Gaussian noise around the class mean is easy for a classifier trained on clean digits to
see through, and its diversity of 0.78 is real. What it cannot do is produce a coherent stroke,
which its coverage of 7.50 (the worst in the table) and novelty of 6.84 measure: each sample is far
from every real image. The generator's samples are individually less "readable" than either
baseline's and collectively much closer to the data.

### Training dynamics

| Epoch | D loss | G loss | D(real) | D(fake) | \|grad G\| | Consistency | Diversity |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5116 | 3.1766 | 0.7834 | 0.1954 | 3.32e+00 | 0.1000 | 0.2735 |
| 5 | 0.7314 | 2.2307 | 0.7431 | 0.2583 | 2.70e+00 | 0.1210 | 0.9262 |
| 10 | 0.6039 | 2.3916 | 0.7916 | 0.2102 | 1.71e+00 | 0.1920 | 1.0215 |
| 13 | 0.6731 | 2.4023 | 0.7743 | 0.2206 | 1.59e+00 | 0.3340 | 1.0071 |
| 15 | 0.7177 | 2.2732 | 0.7607 | 0.2380 | 1.72e+00 | 0.5220 | 0.9374 |
| 18 | 0.7874 | 2.0204 | 0.7317 | 0.2652 | 1.82e+00 | 0.7430 | 0.8839 |
| 20 | 1.0833 | 1.6991 | 0.6703 | 0.3394 | 1.90e+00 | 0.8680 | 0.8557 |
| 25 | 0.8596 | 1.8702 | 0.6979 | 0.2945 | 2.05e+00 | 0.9160 | 0.8394 |
| 30 | 0.8130 | 1.7675 | 0.7179 | 0.2810 | 1.40e+00 | 0.9420 | 0.8787 |

Three things the full table shows.

**The losses do not track progress.** The discriminator's loss goes 0.51 → 0.60 → 1.08 → 0.81 and
the generator's 3.18 → 2.39 → 1.70 → 1.77 while consistency climbs monotonically from 0.10 to
0.94. Between epochs 10 and 13 the generator's loss is essentially flat at 2.39–2.40 and its
consistency nearly doubles. This is why the milestone scores samples with a separate classifier:
each player's loss is measured against an opponent that keeps changing, so neither says how good
the samples are.

**Variety comes first, the label second.** Diversity reaches 1.03 — slightly more varied than the
data — by epoch 9, while consistency is still 0.18. The generator learns to make varied digit-like
images long before it learns which digit each label asks for. Then, as consistency climbs from
0.19 to 0.94 over the next twenty epochs, diversity **falls** to 0.84–0.88. Committing to a class
narrows what the generator produces: this is a mild, measured, within-class collapse, and it is
visible in the figure as rows that are consistent in identity but repetitive in style.

**The label is the slow part.** Twelve epochs pass at near-chance consistency. The discriminator
only benefits from the label once fakes look like digits — before that, fake-ness is obvious from
the pixels alone — and the generator only learns to use the label once the discriminator penalizes
mismatches. At the original DCGAN learning rate of 0.0002 this stage had not begun after 20 epochs:
consistency was 0.147 and the sample grid showed identical rows, which is why the rate was raised
five-fold.

### Part two: the minimax loss

| Epoch | D loss | G loss | D(real) | D(fake) | \|grad G\| | Consistency | Diversity |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6593 | -0.1330 | 0.7621 | 0.2096 | 3.20e-01 | 0.0980 | 0.5447 |
| 5 | 0.6598 | -0.2178 | 0.7698 | 0.2327 | 4.97e-01 | 0.1200 | 0.9864 |
| 10 | 0.5737 | -0.2272 | 0.8051 | 0.1976 | 4.21e-01 | 0.2010 | 1.0479 |
| 15 | 0.5399 | -0.2098 | 0.8112 | 0.1803 | 4.01e-01 | 0.3960 | 1.0141 |
| 18 | 0.6363 | -0.2426 | 0.7876 | 0.2136 | 4.61e-01 | 0.6200 | 0.9597 |
| 19 | 0.8452 | -0.3188 | 0.7628 | 0.2487 | 7.25e-01 | 0.6690 | 0.8972 |
| 20 | **2.6609** | **-1.2432** | **0.5169** | **0.5853** | **2.10e+00** | 0.7130 | 0.9007 |

Over the twenty epochs both runs share, the minimax generator's gradient averaged **5.48e-01**
against **2.37e+00** for the non-saturating one — 4.3x smaller, at every epoch, with `D(fake)` in
the same 0.18–0.25 range for both. That is the saturation the non-saturating loss was designed
around: the two gradients differ by the factor `D(G(z)) / (1 - D(G(z)))`, which is about 1/4 here.

What the run does **not** show is a generator that stalls. By epoch 20 the minimax generator has
reached consistency 0.713 against the non-saturating 0.868 at the same epoch — behind, but far
from stuck. The reason is the optimizer. Adam divides each coordinate's step by a running estimate
of its gradient magnitude, so a gradient that is uniformly four times smaller produces nearly the
same step. The saturation is real in the gradient and mostly hidden in the parameters; under plain
gradient descent the same factor would slow the generator four-fold outright. The gap that
survives Adam comes from the gradient's *direction* being noisier relative to its magnitude, not
from its size.

The last epoch is the other thing worth recording. The discriminator's loss jumps from 0.85 to
**2.66**, its real-sample score falls to 0.52 and its fake-sample score rises to 0.59 — for one
epoch, it is worse than a coin flip, and the generator's gradient norm quadruples. Nothing like
this happens in thirty epochs of the non-saturating run, whose worst discriminator loss is 1.49.
One epoch is one observation, not a trend, and the experiment does not assert it; but it is the
kind of oscillation the minimax objective is known for, and it appeared here without being
looked for.

### Limitations

The digits are speckled: a single-hidden-layer generator with a sigmoid output and 4700 updates is
a small fraction of what a clean MNIST GAN uses, and the comparison between losses and against
baselines is what this measures, not the achievable sample quality. The metrics are distance-based
and judge-based rather than the field's Fréchet or Inception distances, chosen because they can be
computed from scratch and read directly; they agree with each other here but are not comparable to
published figures. And a 10000-image training set with a held-out set of 2000 gives 100 real
images per class for the coverage and diversity references, which is enough to rank the models in
the table and not enough to resolve differences of a few hundredths.
