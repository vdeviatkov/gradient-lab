# Experiment 02: two-layer MLP for XOR

The perceptron milestone ended with a failure it could not fix: XOR is not linearly separable, so
one straight boundary cannot sort its four rows. This milestone adds a hidden layer and shows the
failure go away — and then asks the question the fix invites, which is *how much* hidden layer is
needed. It is implemented twice, in pure Python and in C++, from the same architecture and
hyperparameters.

Both are `2 → h → 1`: tanh hidden units, one linear output, binary cross-entropy with the sigmoid
applied inside the loss, and full-batch gradient descent on gradients written out from the chain
rule. Full batch rather than stochastic, because the dataset is the complete four-row truth table
— there is nothing to sample, and every epoch sees the same gradient, which makes a run a
deterministic function of its seed.

## Running it

Pure Python, no dependencies:

```bash
python -m ml_scratch.pure.xor_mlp_experiment
# or, from the source tree:
python experiments/02_xor_mlp/run.py
# or, after installing the project:
ml-xor-mlp
```

About twelve seconds, nearly all of it the 250 training runs of part two; `--seeds 8` cuts it to
two. It exits unsuccessfully unless the headline network learns XOR, one hidden unit never does,
two hidden units sometimes do and sometimes do not, and four always do.

C++:

```bash
cmake -S . -B build
cmake --build build
./build/cpp_xor_mlp
```

The C++ executable runs the headline case only and exits unsuccessfully unless all four
predictions are correct and the loss reaches its threshold.

## Measured result

Recorded on 2026-10-02 with CPython 3.14.7 and Apple clang 17.0.0 on macOS 26.3.1 (arm64), seed
`7`, learning rate 0.5, stopping loss 0.02. Repeat runs reproduce every number exactly: there is
no shuffling and no sampling, so the only randomness is the initialization the seed fixes.

### Part one: the hidden layer does what the perceptron could not

A `2 → 4 → 1` network, pure Python:

```text
XOR: learned in 344 epochs, binary cross-entropy=0.019932
  (0, 0) -> 0 (p=0.0042, target=0)
  (0, 1) -> 1 (p=0.9770, target=1)
  (1, 0) -> 1 (p=0.9778, target=1)
  (1, 1) -> 0 (p=0.0293, target=0)
```

All four rows correct, where the perceptron on the same truth table never got past three.

The hand-derived gradient agrees with central differences to **9.1e-12** over every parameter.
That check is what makes the derivation in `mlp.py` evidence rather than assertion, and it is the
smallest useful version of what the backpropagation milestone turns into a general facility.

### Part two: how wide the hidden layer has to be

Fifty seeds per width, everything else held fixed:

| Hidden units | Solved | Median epochs | Failures end at |
|---:|---:|---:|---|
| 1 | 0/50 | — | accuracy 0.25, 0.50, 0.75; loss 0.4776, 0.4777, 0.6931 |
| 2 | 29/50 | 451 | accuracy 0.50; loss 0.3468 |
| 3 | 44/50 | 373 | accuracy 0.50; loss 0.3469 |
| 4 | 50/50 | 340 | — |
| 8 | 50/50 | 281 | — |

Three things are worth reading out of that table.

**One hidden unit is still a perceptron.** It never solves XOR, from any of fifty seeds, and
this is not a training failure — it is a representational one. `tanh` is monotone, so
`σ(w · tanh(a·x + b) + c)` is a monotone function of the single linear score `a·x + b`: the
decision boundary is still one straight line, drawn in the same input space, merely squashed
twice on the way out. Forty-five of the fifty failures end at **0.75 accuracy**, three of four
rows — exactly the perceptron's ceiling, arrived at by a different route. Adding depth does not
buy nonlinearity on its own; adding *width* to the hidden layer does, because that is what lets
the output combine more than one boundary.

**Two hidden units is the theoretical minimum and an unreliable one.** Two boundaries suffice to
carve XOR, and the network finds them 29 times in 50. The 21 failures are not scattered: every one
of them lands at accuracy 0.50 and a loss of **0.3468**, which is `(ln 2) / 2` to four decimals.
That is one specific trap, not noise. Its shape is visible in the probabilities — a representative
failure ends at `0.0002, 0.4999, 0.9997, 0.5001`: two rows learned confidently and the other two
left at exactly chance, costing `ln 2` each. It is a plateau rather than an exact stationary
point, with a measured gradient norm of about `2.4e-4`, but flat enough that ten thousand epochs
of full-batch descent never leave it.

**Width buys reliability first and speed second.** Going from 2 to 4 units takes the success rate
from 58% to 100%; going from 4 to 8 leaves it at 100% and cuts the median run from 340 epochs to
281. The first gain is about escaping bad initializations, the second about taking a more direct
path once escaped — and the first is much the larger of the two.

### The two implementations

| | Epochs | Final loss | p(0,0) | p(0,1) | p(1,0) | p(1,1) |
|---|---:|---:|---:|---:|---:|---:|
| pure Python | 344 | 0.019932 | 0.0042 | 0.9770 | 0.9778 | 0.0293 |
| C++ | 376 | 0.019930 | 0.0046 | 0.9799 | 0.9771 | 0.0311 |

Same architecture, same hyperparameters, same seed, same stopping rule — and a 9% difference in
epochs. The reason is initialization and nothing else: both draw Glorot-scaled uniform weights
with zero biases, but `random.Random` and `std::mt19937` with `std::uniform_real_distribution`
produce different numbers from the seed `7`, so the two runs start at different points on the
same loss surface. They end in the same place to four decimals, which is the comparison worth
making. `docs/design/reproducibility.md` states this expectation in advance; this is the
measurement of it.

## Limitations

Fifty seeds per width is enough to separate 0/50 from 29/50 from 50/50 and not enough to put a
tight interval on the 58%. The sweep varies width at one learning rate and one stopping loss, so
"two units fail 42% of the time" is a statement about this configuration — a smaller step or a
longer budget would change the number, though not the ordering. The plateau at `(ln 2) / 2` is
identified by its loss, accuracy, probabilities and gradient norm, not by characterizing the
weights that produce it. And the representational claim about one hidden unit is proved from the
monotonicity of `tanh` and `σ`; the fifty seeds only confirm it.
