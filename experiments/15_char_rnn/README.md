# Experiment 15: a character-level recurrent network

A tanh recurrent network trained to predict the next character of Shakespeare by truncated
backpropagation through time, scored against count-based n-gram models on fixed held-out text,
and then taken apart: how far its gradient reaches back in time, what the truncation window
costs, and what gradient clipping does when the optimizer takes the gradient at face value.

## Getting the data

Tiny Shakespeare is not committed. Download it first:

```bash
./scripts/download_tiny_shakespeare.sh
```

`ML_SCRATCH_TINY_SHAKESPEARE` overrides the file's location.

## Setup

**The corpus.** The first 500000 characters of the file, split contiguously into 450000 training,
25000 validation, and 25000 test characters, in that order, with a vocabulary of the 63 distinct
characters that occur. The splits are contiguous because a random split would let a sequence
model see the characters on either side of every held-out one. The LSTM, GRU, and Transformer
milestones read exactly this prefix, split, and vocabulary, so their results are comparable to
this one.

**The network.** One-hot input, 128 tanh hidden units, 63 output logits: 32703 parameters.
Truncated BPTT over windows of 50 steps, with the training text cut into 32 contiguous streams
whose hidden states carry across windows; Adam at 0.002; gradient norm clipped at 5; 12 epochs;
seed `20260919`.

**The baselines.** Unigram, bigram, and trigram models fit on the training split with add-one
smoothing. Every model is scored by mean cross-entropy per predicted character in nats on the
same validation and test splits, with bits per character and perplexity alongside.

**The measurements.**

- *Gradient reach*: for the last prediction of a 60-character window, the norm of its gradient
  with respect to the hidden state `lag` steps earlier, relative to lag 0, averaged over 20
  windows of the validation text; before and after training.
- *Part two, the truncation window*: the same network from the same initial parameters, trained
  for 12 epochs with windows of 5 and of 1 instead of 50. A shorter window makes proportionally
  more updates per epoch, so the runs are compared both at equal epochs and at equal update
  counts.
- *Part three, clipping*: the same network under plain gradient descent at 0.7, four epochs each,
  with no clipping, clipping at 5, and clipping at 1. Adam scales every step by a running
  gradient magnitude, which hides what clipping does; plain gradient descent does not.

## Running it

```bash
cmake -S . -B build
cmake --build build
./build/cpp_char_rnn
```

About six minutes. It writes a checkpoint to `data/char_rnn.checkpoint`
(`ML_SCRATCH_CHAR_RNN_CHECKPOINT` overrides it) and prints two seeded samples. It exits
unsuccessfully unless the network beats every n-gram on test, its validation loss is still falling
at the last epoch, test and validation agree within 0.2 nats, the gradient reaching 20 steps back
is below a tenth of what reaches the last state, the one-step window is the worst of the three at
equal epochs, the 50-step window beats the 5-step one at equal update counts, the unclipped
gradient-descent run's largest gradient norm exceeds ten times the tight threshold, and each
tighter clipping threshold ends with a lower validation loss than the looser one.

## Measured result

Recorded on 2026-09-19 with Apple clang 17.0.0 on macOS 26.3.1 (arm64), CMake build type `Release`,
seed `20260919`. A repeat run reproduced every loss, gradient norm, and sample exactly; characters
per second and wall-clock seconds are the columns that move. Total wall time 5 minutes 46 seconds on an otherwise idle machine.

### Against the baselines

| Model | Parameters | Validation | bits/char | Perplexity | Test | bits/char | Perplexity |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform guess | 0 | 4.1431 | 5.977 | 63.0 | 4.1431 | 5.977 | 63.0 |
| unigram | 63 | 3.3007 | 4.762 | 27.1 | 3.2810 | 4.734 | 26.6 |
| bigram | 3969 | 2.5250 | 3.643 | 12.5 | 2.5181 | 3.633 | 12.4 |
| trigram | 250047 | 2.1460 | 3.096 | 8.55 | 2.1536 | 3.107 | 8.62 |
| untrained RNN | 32703 | 4.1750 | 6.023 | 65.0 | 4.1730 | 6.020 | 64.9 |
| **trained RNN** | 32703 | **1.9324** | **2.788** | **6.91** | **1.9637** | **2.833** | **7.13** |

The network scores **1.96 nats on test**, 0.19 below the trigram with an eighth of the trigram's
parameters, and its validation loss is still falling at epoch 12 (1.9435 → 1.9324 over the last
epoch). Test and validation agree within 0.03 nats. Training ran at **68600 characters per second**,
81 seconds for 12 epochs.

Two things to read from the baselines. The trigram is not weak: it covers 2.0 of the 2.2 nats
between the uniform guess and the network on test, and any claim that a sequence model "learns
language" has to clear it. And the untrained network is slightly *worse* than the uniform guess
(4.175 against 4.143), which is what random logits do; every nat below 4.14 is learned.

### Training

| Epoch | Train loss | Validation | bits/char | Clipped | Max \|grad\| | Mean \|grad\| |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.5613 | 2.3113 | 3.335 | 0.0000 | 2.91 | 0.63 |
| 2 | 2.1042 | 2.2093 | 3.187 | 0.0000 | 3.72 | 0.63 |
| 4 | 1.9068 | 2.0969 | 3.025 | 0.0000 | 3.19 | 0.69 |
| 6 | 1.7999 | 2.0278 | 2.926 | 0.0000 | 1.80 | 0.72 |
| 8 | 1.7325 | 1.9859 | 2.865 | 0.0000 | 1.53 | 0.70 |
| 10 | 1.6858 | 1.9541 | 2.819 | 0.0000 | 1.44 | 0.70 |
| 12 | 1.6517 | 1.9324 | 2.788 | 0.0000 | 1.29 | 0.71 |

**Clipping never fired.** The largest gradient norm in 3372 updates was 3.72 against a threshold of
5, and the mean sat at 0.7 throughout. Under Adam this is unsurprising and is why part three
exists: Adam divides each coordinate's step by a running estimate of its gradient magnitude, so a
gradient twice as large as usual produces a step barely larger than usual, and the network never
gets pushed far enough for its recurrence to blow up.

### Samples

Seeded, 300 characters after the prompt `ROMEO:`.

At temperature 0.5:

```text
ROMEO:
Nor he slouth a mint
With strath of the course to batt we like the bears of the connuspart to seed the could be shish and and to the lovery to be the meant to will be all say.

CORIOLANUS:
Now I sme, the curse to mane us must he wear with the will as seet him stouls is double do to hearn to with a t
```

At temperature 1.0:

```text
ROMEO:
Meanen diuswer and wat, I, us wirk, our dister sur 'twarend hid speak not genbice;
My sent the but men's, I'll so; what had me. Inle misforthou for and lends,
Your croble fait
We to but defirim
As if when youm myinclans, poot he we have open to ful to a wange it with siepl
As I grough our way? I bes
```

The structure of the format is learned — speaker names in capitals followed by a colon and a
newline, line lengths, punctuation, a blank line between speeches — and so is the shape of English
words: short function words are real, longer words are pronounceable inventions. Nothing beyond a
few words is coherent, which is what a 128-unit network with the gradient reach below is capable
of. The low temperature produces more real words and more repetition; the high temperature more
variety and more invention.

### Gradient reach

|dL_T/dh_{T-lag}| relative to lag 0, mean over 20 validation windows:

| | lag 0 | 1 | 2 | 5 | 10 | 20 | 30 | 50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| before training | 1.00 | 0.970 | 0.954 | 0.740 | 0.530 | 0.269 | 0.127 | 3.1e-02 |
| after training | 1.00 | 0.841 | 0.634 | 0.243 | 0.068 | 5.5e-03 | 7.0e-04 | 1.6e-05 |

This is the vanishing gradient measured directly. After training, the gradient that reaches ten
steps back is **7% of what reaches the last state**, twenty steps back it is **0.6%**, and fifty
steps back it is essentially zero — inside a window that is nominally 50 steps long. The network's
effective context is about ten characters, and the samples read that way.

Training made it *worse*, not better: at lag 20 the untrained network kept 27% of the gradient
and the trained one 0.6%. The product of Jacobians contains a `1 - h²` factor per step, and a
trained network drives its hidden units toward saturation, where that factor is small. Nothing in
the objective rewards keeping the gradient alive across time, so nothing does. The gated
architectures of the next milestone exist to change that.

### Part two: the truncation window

Same initial parameters, 12 epochs each:

| Window | Updates per epoch | Validation, epoch 1 | Epoch 4 | Epoch 8 | Epoch 12 | Mean \|grad\| |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 281 | 2.3113 | 2.0969 | 1.9859 | 1.9324 | 0.70 |
| 5 | 2812 | 2.1139 | 1.9304 | 1.8880 | **1.8790** | 1.14 |
| 1 | 14062 | 2.1309 | 2.0360 | 2.0133 | 1.9954 | 1.79 |

The result cuts two ways, and both halves are the point.

**At equal epochs the 5-step window wins**, 1.879 against 1.932, and it is not close. A window of
5 makes ten times as many updates per pass over the text, each from a tenth as many predictions,
and on this budget that trade is worth more than the extra 45 steps of reach — which, per the
table above, the network was barely using anyway. The 5-step run is also nearly flat by epoch 12
(1.8789 → 1.8790 over the last two), while the 50-step run is still improving by 0.011 per epoch;
with a larger budget the order would likely reverse, and this experiment does not have that
budget.

**At equal update counts the 50-step window wins**, 1.954 after 10 epochs against 2.114 for the
5-step window after 1 epoch, both at 2812 updates. Each of the longer window's updates carries
gradient from ten times as much context, and per update that is worth 0.16 nats.

**The 1-step window is the worst of the three** at every epoch after the first — 1.995 at epoch
12 — despite making fifty times the updates of the main run. With a window of one, no gradient
ever crosses a step boundary: the recurrence is trained only to make the *next* character
predictable from whatever state it happens to be in, and never to store anything that pays off
later. It still beats the trigram (2.146), because the carried state does hold some context even
when nothing trained it to; the unit test on a delayed-copy stream shows where that runs out.

### Part three: clipping under plain gradient descent

Gradient descent at 0.7, four epochs each, same initial parameters:

| Clipping | Epoch 1 max \|grad\| | Clipped, epoch 1 | Validation, epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 |
|---|---:|---:|---:|---:|---:|---:|
| none | 37.19 | — | 4.5202 | 4.6264 | 3.8614 | **5.5743** |
| at 5 | 33.69 | 38.1% | 3.4466 | 2.9235 | 2.6204 | 2.5182 |
| at 1 | 2.78 | 9.6% | 2.4515 | 2.3410 | 2.2642 | **2.2062** |

Here the explosion is real. Without clipping, the largest gradient norm in the first epoch is
**37**, fifty times the mean of the Adam run, and the parameters are thrown far enough that the
network never recovers: its validation loss after four epochs is 5.57, *worse than the uniform
guess*. Clipping at 5 catches it — 38% of the first epoch's updates are clipped, the mean norm
falls from 4.1 to 1.0 by the second epoch as the network settles, and it reaches 2.52. Clipping
at 1 is better still, 2.21, and its largest first-epoch norm is 2.78: with the step bounded from
the very first update, the parameters never enter the region where the gradient blows up in the
first place.

Read against part one, the lesson is about optimizers as much as about clipping. The same network
on the same text under Adam had a largest norm of 3.7 and needed no clipping; under plain gradient
descent it needed clipping to train at all. Clipping is the guard for an optimizer that scales its
step by the gradient, and Adam is not one.

### Limitations

A single-layer tanh network with 128 units is the smallest recurrent model that produces
recognizable text, and 1.96 nats is far from what a gated or attention-based model reaches on this
corpus; the comparison between windows and between clipping settings is what this measures, not
the achievable loss. The 500000-character prefix is a fifth of the corpus, chosen so that the
three parts fit in minutes; the sequence milestones that follow use the same prefix so they can be
compared with this one. And the equal-update comparison in part two lands on one pair of epochs
(10 and 1) because the update counts only coincide there; it is one observation, though a large
one.
