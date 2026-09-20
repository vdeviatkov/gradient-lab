# Experiment 16: LSTM and GRU cells against the elman cell

The recurrent milestone left one measurement pointing at its own limit: the gradient reaching
twenty steps back was 0.6% of the gradient at the last step. This experiment swaps the cell — an
LSTM, a GRU — inside exactly the same training loop, corpus, optimizer, window and streams, and
measures what the gates change: the loss on text at equal parameters and at equal width, how far
the gradient reaches, and whether a dependency the elman cell cannot learn becomes learnable.

## Getting the data

Tiny Shakespeare is not committed. Download it first:

```bash
./scripts/download_tiny_shakespeare.sh
```

`ML_SCRATCH_TINY_SHAKESPEARE` overrides the file's location.

## Setup

Everything experiment 15 fixed is kept: the first 500000 characters of the corpus, the contiguous
450000 / 25000 / 25000 split, the 63-character vocabulary, one-hot input, truncated BPTT over
50-step windows with 32 carried streams, Adam at 0.002, clipping at 5, seed `20260919`. The elman
run is experiment 15's main run repeated, and its every number matches.

**Part one, equal parameters.** Per hidden unit an LSTM has four gate rows and a GRU three
against the elman's one, so at a fixed budget a gated cell is narrower: 128 elman units (32703
parameters), 58 LSTM units (32021), 70 GRU units (32613), 12 epochs each.

**Part two, equal width.** LSTM and GRU at 128 units, 4 epochs each, against the elman run at its
fourth epoch. A gated cell of that width costs three to four times the arithmetic per character,
which is why this comparison is shorter.

**Gradient reach.** For each part-one network after training, the norm of the last prediction's
gradient with respect to the carried state `lag` steps earlier — h and c together for the LSTM —
relative to lag 0, averaged over 20 windows of the validation text.

**Part three, a delayed-copy stream.** Every other character is a copy of the random bit written
`delay` pairs earlier, so the copy at index 2i+1 repeats the bit at index 2(i−delay), which
entered the network 2·delay steps before the copy is predicted. The random bits are
unpredictable, so the best possible loss is ln 2 / 2 = 0.347 and chance is ln 2 = 0.693. Each
cell at 32 units, window 40, Adam at 0.01, 60 epochs, at delays of 3, 6 and 10. After training,
each network's gradient reach on its own stream is measured at the lag of the source bit.

## Running it

```bash
cmake -S . -B build
cmake --build build
./build/cpp_lstm_gru
```

About six and a half minutes. It exits unsuccessfully unless the elman run reproduces experiment
15's final validation loss, the GRU beats the elman cell on test at equal parameters and the LSTM
does not, both gated cells beat it at equal width, the GRU's reach at lag 20 on text exceeds the
elman's tenfold, every cell learns the copy at six steps, at twelve steps the elman cell scores
chance while both gated cells reach the floor, at twenty steps the elman cell still scores chance
while the LSTM reaches the floor, and the LSTM's reach to the source bit at twenty steps exceeds
the elman's tenfold.

## Measured result

Recorded on 2026-09-19 with Apple clang 17.0.0 on macOS 26.3.1 (arm64), CMake build type `Release`,
seed `20260919`. A repeat run reproduced every loss and reach exactly; characters per second and
wall-clock seconds are the columns that move. Total wall time 6 minutes 19 seconds.

### Part one: equal parameters

| Cell | Units | Parameters | Validation | Test | bits/char | Perplexity | chars/s | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| elman | 128 | 32703 | 1.9324 | 1.9637 | 2.833 | 7.13 | 69200 | 80.3 |
| LSTM | 58 | 32021 | 2.0238 | **2.0514** | 2.960 | 7.78 | 86400 | 64.4 |
| **GRU** | 70 | 32613 | **1.8789** | **1.9192** | 2.769 | 6.82 | 83500 | 66.5 |

Per-epoch validation loss:

| Epoch | elman | LSTM | GRU |
|---:|---:|---:|---:|
| 1 | 2.3113 | 2.5069 | 2.3221 |
| 2 | 2.2093 | 2.3082 | 2.1630 |
| 4 | 2.0969 | 2.1802 | 2.0235 |
| 6 | 2.0278 | 2.1182 | 1.9618 |
| 8 | 1.9859 | 2.0791 | 1.9227 |
| 10 | 1.9541 | 2.0459 | 1.8974 |
| 12 | 1.9324 | 2.0238 | 1.8789 |

At the same parameter count, **the GRU beats the elman cell and the LSTM loses to it** — and the
ranking holds at every epoch, not just the last. The GRU is 0.045 nats better on test with 70
units; the LSTM is 0.088 nats worse with 58. Both gated cells are still improving at epoch 12 (the
LSTM by 0.010 per epoch, the elman by 0.011), so the gap is not a matter of the LSTM having
converged lower; its training loss is also the highest of the three (1.7787 against 1.6517 and
1.5834), which says it is fitting less, not generalizing worse.

Fifty-eight units is the price of four gates. The LSTM spends its budget on machinery for keeping
state, and on this text — where experiment 15 found a 5-step window enough to beat a 50-step one —
that machinery is worth less than the 70 extra units the elman cell gets instead. The GRU, with
three gate rows, keeps enough width to come out ahead. Neither result is the textbook ordering,
and both are what the run measured.

The gated cells are also *faster* per character here (83–86k against 69k), because their smaller
hidden vector shrinks the output layer, which at 63 outputs is a large share of the arithmetic.

### Part two: equal width

| Cell | Units | Parameters | Validation, epoch 4 | Test | chars/s | Seconds |
|---|---:|---:|---:|---:|---:|---:|
| elman (from part one) | 128 | 32703 | 2.0969 | — | 69200 | — |
| LSTM | 128 | 106431 | 2.0836 | 2.1075 | 20800 | 89.2 |
| GRU | 128 | 81855 | **1.9593** | **1.9672** | 28900 | 64.1 |

At equal width both gated cells beat the elman cell at epoch 4: the GRU by 0.138 nats, the LSTM by
0.013. The GRU at 128 units after four epochs (1.959) is within 0.027 of where the elman cell
finished after twelve (1.932). The costs are in the table: 3.3 times the parameters and 3.4 times
the time per character for the LSTM, 2.5 and 2.4 times for the GRU.

One row of the full output deserves a note. The 128-unit LSTM's validation loss after its first
epoch was **5.49** — worse than the uniform guess of 4.14 — while its training loss was 2.77, and
by the second epoch it was 2.24. Tracing it showed a learned bistability: from the zero state,
certain first characters (a capital letter, a newline) put the cell into a regime where its cell
state never grew, and the output layer, calibrated for the large-cell-state regime the training
streams live in, was confidently wrong from then on. Every training stream starts from the zero
state at a fixed position, so the network had seen the zero state only 32 times per epoch. One
more epoch removed it. It is recorded here because it is exactly the failure a persistent state
makes possible: a wrong state, once entered, is *kept*.

### Gradient reach on text

|dL_T/d(state)_{T−lag}| relative to lag 0, mean over 20 validation windows, part-one networks:

| Cell | lag 1 | 2 | 5 | 10 | 20 | 30 | 50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| elman | 0.841 | 0.634 | 0.243 | 0.068 | 5.5e-03 | 7.0e-04 | 1.6e-05 |
| LSTM | 0.806 | 0.528 | 0.148 | 0.021 | 1.8e-03 | 2.9e-04 | 1.5e-05 |
| GRU | 0.766 | 0.595 | 0.223 | **0.120** | **7.8e-02** | **1.9e-02** | **1.8e-03** |

The GRU carries fourteen times the gradient to twenty steps back that the elman cell does, and a
hundred times at fifty. **The LSTM carries less than the elman cell.** Its additive path exists,
but nothing on this text made it keep the forget gates open, and a trained LSTM forgets as fast as
the task lets it. Reach is a property of the trained network on its data, not of the cell alone;
part three shows the same LSTM cell with a very different profile.

### Sample

From the GRU with 70 units, temperature 0.5, seeded, after the prompt `ROMEO:`:

```text
ROMEO:
I have herest restit unthoush with inmonself,
And have some of the beding held and sword, I know.

GLOUCESTER:
Stay, and be of your consest to be the peace there
That I than the love the mistreach seet me when the common;
This have here the brother the cunces on the heads,
And so, so for the stand t
```

More real words and better line structure than experiment 15's elman sample at the same
temperature, with the same limits: nothing coheres past a phrase.

### Part three: the delayed-copy stream

Loss after 60 epochs (floor 0.347, chance 0.693), and each trained network's gradient reach on its
own stream at the lag of the source bit:

| Delay | Steps back | elman | LSTM | GRU | Reach: elman | LSTM | GRU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 6 | 0.3456 | 0.3447 | 0.3560 | 4.0e-01 | 2.7e-01 | 3.4e-01 |
| 6 | 12 | **0.6960** | 0.3387 | 0.3448 | 3.4e-03 | **1.46** | 4.0e-01 |
| 10 | 20 | **0.6942** | **0.3667** | **0.6932** | 7.1e-04 | **1.7e-01** | 6.0e-05 |

At six steps every cell learns the copy. At twelve, the elman cell scores exactly chance — it has
not learned that the copies are copies — while both gated cells reach the floor. At twenty, only
the LSTM learns it; the GRU at this budget does not.

The reach column is the mechanism, measured. The elman cell that failed at twelve steps carries
0.3% of the gradient to the source bit; the LSTM that succeeded carries **146%** — the gradient
*grows* along its cell path, because the forget gates it learned are near one and the product of
them exceeds what the output side loses. At twenty steps the LSTM still carries 17% while the
elman carries 0.07% and the GRU, which did not learn the task, 0.006%. Compare the GRU's two
rows with its behaviour on text, where it had the longest reach of the three: the same cell reaches
far when the data rewarded it and not otherwise.

### Reading the parts together

Three comparisons and three different winners. At equal parameters the GRU wins on text and the
LSTM loses; at equal width both gated cells win; on a task that genuinely needs twenty steps of
memory only the LSTM succeeds. There is no contradiction. Gates buy the *ability* to carry a
gradient through many steps, at a cost in units or arithmetic; whether that ability is worth its
price depends on whether the data has dependencies long enough to use it, and Shakespeare at the
character level mostly does not. The delayed-copy stream does, and there the ordering is the
textbook one.

### Limitations

Twelve epochs on a 450000-character prefix is a small budget, and the part-one ordering could
change with more of either; the LSTM's training loss was still falling fastest of the three at
epoch 12. Part two's four epochs compare convergence speed at equal width more than final quality.
The copy stream uses one seed and one hidden size, and the GRU's failure at twenty steps is a
failure at this budget, not a bound. And gradient reach is measured on the trained network's own
data with the loss of a single prediction, which is the quantity BPTT actually propagates but is a
relative measure: a reach of 1.46 means the gradient grew along the path, not that it is large in
absolute terms.
