# Experiment 17: a decoder-only Transformer

A small Transformer — sinusoidal positions, causal multi-head self-attention, pre-normalized
residual blocks, every gradient derived by hand — trained on the same corpus, splits and
vocabulary as the recurrent milestones, and read against them: the loss at the same epochs and
at the same parameter count, what each of its heads attends to, which parts of the architecture
its loss depends on, and whether a twenty-step copy that the elman cell could never learn is
one lookup away.

## Getting the data

```bash
./scripts/download_tiny_shakespeare.sh
```

`ML_SCRATCH_TINY_SHAKESPEARE` overrides the file's location.

## Setup

The first 500000 characters, 450000 / 25000 / 25000 contiguous, 63 characters, seed `20260921`.
Every model trains on random windows of `context + 1` characters, 32 per update, Adam, gradient
norm clipped at 1; an epoch is one training text's worth of characters. Unlike the recurrent
networks there is no carried state, so validation and test are scored in abutting windows: a
character early in a window is predicted from only the few before it.

**Part one.** Two blocks of 4-head attention with a 256-wide feed-forward layer, width 64,
context 64, Adam at 0.003, 8 epochs — 107711 parameters, comparable to experiment 16's 128-unit
LSTM (106431) and GRU (81855). Its validation loss after each epoch is printed next to
experiment 15's 128-unit elman cell and experiment 16's 70-unit GRU on the same corpus.

**Where the heads look.** For each head, the mean attention weight it places at each distance
behind the query, from the last position of 50 validation windows: summarized as the mean
distance, the weight within three characters, and the single most attended distance.

**Part two.** Width 32 — 29311 parameters, the recurrent networks' budget — for 6 epochs, and four
ablations from the same initialization and budget: no positional encoding; a learned relative
position bias added to the sinusoidal encoding; one head instead of four; a context of 16 instead
of 64.

**Part three.** Experiment 16's delayed-copy stream with the copies written as 2 and 3 instead of
0 and 1, so the current token says whether a copy comes next. A recurrent cell tracks that
parity in its state for free; a Transformer would have to read it off the positional encoding,
and a sinusoid has no period-two component to read it from — an earlier attempt on the unmarked
stream plateaued at exactly the loss of attending to the right place without knowing which
positions are copies. The three recurrent cells (32 units, window 40, 60 epochs) and four
Transformers (width 16, one block, two heads, context 40, 30 epochs, with and without the
sinusoidal encoding and the relative bias) are trained on the marked stream. The Transformers
are scored with a stride of 20, so every prediction after the first window has the source bit in
view; the recurrent cells carry their state. The floor is ln 2 / 2 = 0.347 for all of them.

## Running it

```bash
cmake -S . -B build
cmake --build build
./build/cpp_transformer
```

About seven and a half minutes. It writes `data/char_transformer.checkpoint`
(`ML_SCRATCH_TRANSFORMER_CHECKPOINT` overrides it). It exits unsuccessfully unless the main model
beats the trigram and the elman cell on test, its validation loss is lowest at its last epoch, the
model without a positional encoding is worse by more than 0.2 nats, the relative bias helps, the
sinusoidal Transformer reaches the copy floor within 0.03 with some head putting more than half
its weight on the source bit, the Transformer without positions is worse on the copy by more than
0.1, and the elman cell scores chance on it.

## Measured result

Recorded on 2026-09-21 with Apple clang 17.0.0 on macOS 26.3.1 (arm64), CMake build type `Release`,
seed `20260921`. A repeat run reproduced every loss and attention weight exactly; characters per
second and wall-clock seconds are the columns that move. Total wall time 7 minutes 19 seconds on the first run and 6 minutes 58 seconds on the repeat.

### Part one: against the recurrent networks

| Epoch | Transformer 64 | elman 128 | GRU 70 |
|---:|---:|---:|---:|
| 1 | 2.3360 | 2.3113 | 2.3221 |
| 2 | 2.1617 | 2.2093 | 2.1630 |
| 3 | 2.0712 | 2.1487 | 2.0755 |
| 4 | 2.0140 | 2.0969 | 2.0235 |
| 5 | 1.9575 | 2.0590 | 1.9886 |
| 6 | 1.9643 | 2.0278 | 1.9618 |
| 7 | 1.9217 | 2.0065 | 1.9402 |
| 8 | **1.9091** | 1.9859 | 1.9227 |

| Model | Parameters | Validation | Test | bits/char | Perplexity | chars/s | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Transformer, 8 epochs | 107711 | 1.9091 | **1.9350** | 2.792 | 6.92 | 18300 | 202 |
| elman 128, 12 epochs (exp. 15) | 32703 | 1.9324 | 1.9637 | 2.833 | 7.13 | 69200 | 80 |
| GRU 70, 12 epochs (exp. 16) | 32613 | 1.8789 | 1.9192 | 2.769 | 6.82 | 83500 | 67 |
| trigram (exp. 15) | 250047 | 2.1460 | 2.1536 | 3.107 | 8.62 | — | — |

After the first epoch the Transformer is ahead of both recurrent networks at every epoch — by
0.077 nats over the elman cell and 0.014 over the GRU at epoch 8 — and its test loss after 8
epochs, **1.935**, beats the elman cell's after 12. It does not beat the GRU's 12-epoch figure
(1.919), and it costs 3.3 times the parameters and 3.8 times the time per character to get
there. The claim that attention beats recurrence on this corpus at this budget is therefore
partly true: per epoch it learns faster than either; per parameter and per second it does not.
Its own validation loss is still falling by 0.013 per epoch at the end.

Note the epoch-6 blip (1.9575 → 1.9643 → 1.9217): with random windows every epoch is a different
sample of the text, and the validation curve is not monotone the way the carried-state
recurrent curves were.

### Where the heads look

| Block | Head | Mean distance | Within 3 | Favourite |
|---:|---:|---:|---:|---:|
| 1 | 1 | 2.1 | 0.946 | 1 |
| 1 | 2 | **29.9** | 0.052 | 21 |
| 1 | 3 | 1.1 | 0.971 | 1 |
| 1 | 4 | 7.4 | 0.353 | 4 |
| 2 | 1 | 11.2 | 0.325 | 0 |
| 2 | 2 | 3.5 | 0.779 | 1 |
| 2 | 3 | 2.4 | 0.763 | 1 |
| 2 | 4 | 3.5 | 0.641 | 1 |

Six of the eight heads are local: they put two thirds to 97% of their weight within three
characters, mostly on the previous one. One head in the first block (head 2) is the opposite —
its weight is spread across the whole window with a mean distance of 30 and almost nothing
nearby — and one (block 1, head 4) sits in between at a mean of 7. This is the division of
labour attention makes possible and a recurrent state does not: most of the model's capacity
attends to the immediate context that experiment 15 showed carries most of the signal, and one
head is free to look far back without paying for it in the others. The far-looking head's
"favourite" of 21 is not a fixed offset — its weight is diffuse — but its existence is what lets
the model use context the 5-step window of experiment 15 could not.

### Sample

Temperature 0.5, seeded, after `ROMEO:`:

```text
ROMEO:
Who conce him to know hold from the comes too sently lords.

MENENIUS:
I the made have shame to the coul for was the will of me,
And for ment, that with have for from unchesed
So dissure name carn not have to me cossess
If the for to the the part the to death to the wars our them.

RATCLIFF:
Have wa
```

The same character-level texture as the recurrent samples — real short words, invented long ones,
the play's format intact — at a slightly lower loss.

### Part two: the recurrent networks' budget, and the ablations

Width 32, 6 epochs each, same initialization:

| Model | Parameters | Validation, epoch 6 | Test | chars/s |
|---|---:|---:|---:|---:|
| width 32, 4 heads, context 64 | 29311 | 2.0808 | 2.0798 | 63000 |
| no positional encoding | 29311 | **2.4104** | 2.4215 | 63700* |
| plus relative position bias | 29823 | **2.0099** | 2.0071 | 63200 |
| one head | 29311 | 2.0563 | 2.0556 | 71800 |
| context 16 | 29311 | 2.0291 | 2.0430 | 75200 |
| elman 128 at epoch 6 (exp. 15) | 32703 | 2.0278 | — | 69200 |
| GRU 70 at epoch 6 (exp. 16) | 32613 | 1.9618 | — | 83500 |

\* On the first run this row's throughput fell from 63k to 39k characters per second after its
first epoch and stayed there; the repeat run held 63.7k throughout with every loss identical, so
the figure shown is the repeat's and the dip was machine noise.

At the recurrent networks' parameter count the Transformer is **behind both** at epoch 6 — 2.081
against the elman cell's 2.028 and the GRU's 1.962. Attention's overhead (four projections per
block plus a feed-forward layer four times the width) leaves a 32-wide model with little
capacity per parameter, and at this size the recurrent cells spend theirs better. The part-one
advantage came from width, not from architecture alone.

The ablations sort into what the architecture needs and what, at this size, it does not:

- **Positions are essential.** Without the encoding the loss is worse by 0.33 nats and the
  model barely improves after its second epoch (2.46 → 2.41 over four epochs). With the causal
  mask and two blocks it is not entirely order-blind — the test suite shows depth leaks order —
  but what leaks is not enough to use.
- **Telling heads the distance helps.** Adding a learned relative position bias — 512 extra
  parameters, one per head per distance per block — improves the loss by 0.071 nats, the largest
  gain of any change here. Sinusoidal positions say where a token *is*; a head that wants the
  previous character has to compute "one behind me" from two absolute positions every time, and
  a bias that says so directly is cheaper to learn.
- **One head beat four**, 2.056 against 2.081. With width 32, four heads are four 8-dimensional
  attentions; one head has the full 32. Part one's profile showed most heads doing the same local
  job, and at this width splitting the dot product into four small ones costs more than the
  variety buys.
- **Context 16 beat context 64**, 2.029 against 2.081. This is experiment 15's truncation result
  again in a different form: the same characters seen per epoch in windows a quarter the length
  make four times the updates, and on this corpus the context beyond 16 characters is worth less
  than the extra updates — at least until the longer-context model catches up, which six epochs
  did not give it time to do.

The two surprises are both about budget, not about attention. Neither is asserted by the
experiment; both are what it measured.

### Part three: the marked copy stream

| Model | Loss | Best head's weight on the source bit |
|---|---:|---:|
| elman, 32 units | 0.6960 | — |
| LSTM, 32 units | 0.6942 | — |
| GRU, 32 units | 0.3823 | — |
| Transformer, sinusoidal | **0.3486** | **0.814** |
| Transformer, relative bias only | 0.3487 | 0.300 |
| Transformer, both | 0.3486 | 0.633 |
| Transformer, neither | 0.6825 | 0.033 |

Floor 0.347. The Transformer with any positional information solves the twenty-step copy to
within 0.002 of the floor in 30 epochs, and the head that does it puts 81% of its weight on the
bit twenty steps back. Without positional information it cannot — its loss is at chance on the
copies and its best head puts 3% on the source — because attention over an unordered past has
no way to name "twenty steps ago". This is the architectural claim, measured: a dependency the
recurrent state had to carry through twenty multiplications is one attention weight.

The recurrent rows are the same cells at the same size as experiment 16, on the marked stream.
The elman cell still cannot learn it. The GRU gets most of the way (0.382), where on the
unmarked stream it did not; the LSTM, which learned the unmarked twenty-step copy in experiment
16, scores chance here. The marker changes the task's difficulty for a recurrent cell in a way
that is not obviously monotone, and these single-seed results are reported as they came out
rather than explained.

The relative-bias-only model reaches the floor with its best head at only 30% on the source: with
a bias per distance and no encoding, several heads and positions share the work, and the loss
does not care how.

### Limitations

Eight epochs of a 107711-parameter model and six of a 29311-parameter one are small budgets, and
every comparison here is at a fixed budget; the part-two ranking in particular (one head, short
context) is a statement about six epochs at width 32. The recurrent references are quoted from
experiments 15 and 16 rather than retrained, which is fair because the corpus, splits and
vocabulary are identical and the earlier runs reproduce exactly, but the transformer was tuned
(learning rate 0.003, clipping at 1) separately. Evaluation in abutting windows understates the
Transformer slightly, since a character early in a window has less context than it would in a
sliding evaluation, and the recurrent networks' carried state has no such handicap. And the copy
stream is one seed and one delay.
