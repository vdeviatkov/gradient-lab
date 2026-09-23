# Experiment 19: implementation benchmarks

Every earlier milestone quoted a wall-clock column next to its accuracy. This one measures those
costs properly and in one place: a benchmarking harness that auto-tunes its iteration count,
separates the cold first call from the steady state, reports how much its own repetitions
disagree, and records the machine and build alongside the numbers — plus the memory the process
holds and the size of the code that produced them.

The result is also the format the other language implementations will be compared in: the run
writes `results/benchmarks/cpp.json` with every repetition kept, so a summary can be recomputed
rather than taken on trust.

## What the harness does

- **Auto-tuning.** A body faster than the clock can resolve is run many times per repetition,
  chosen so one repetition reaches a target duration. The tabular Q-learning update takes 1.9 ns
  and is measured over 31 million iterations; the convolutional gradient takes 112 µs and is
  measured over 14.
- **Cold versus steady state.** The very first call is timed before anything has been warmed, and
  reported as a multiple of the steady-state median. That gap is what a warm-up hides, and it is
  large: up to 88× here.
- **Repetitions, not one shot.** Seven timed repetitions; the figure is the median of their means,
  because one repetition interrupted by the operating system moves a mean and leaves a median
  alone. The standard deviation across repetitions is reported as a fraction of the mean, and the
  run **fails** if any benchmark's repetitions disagree by more than 20%.
- **A reset callback** that runs outside the timed region, so a benchmark that consumes state can
  restore it without being charged.
- **`do_not_optimize`**, because a loop whose result is never read is dead code a compiler may
  delete; a test in the suite checks that four thousand square roots really are twenty times
  slower than an empty body.
- **Environment capture** — compiler and version, language standard, build type (passed in via
  `ML_SCRATCH_BUILD_TYPE`, since the source cannot see it), platform, architecture, pointer width,
  date — and **peak resident memory** where the platform offers it.
- **Clock characterization.** The resolution and the cost of one reading are measured rather than
  assumed, and every timed repetition here is over a million times the resolution.

Compilation time is the one cost the benchmarked program cannot measure about itself, since by
the time it runs it has already been built. `scripts/benchmark_build.sh` times a clean build
instead and writes `results/benchmarks/cpp_build.json`.

## Running it

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ML_SCRATCH_BUILD_TYPE=Release ./build/cpp_benchmarks

./scripts/benchmark_build.sh   # compilation timing, written separately
```

About twelve seconds, and **it must be run on an idle machine** — if it is not, the spread check
fails and says so rather than publishing the numbers. It needs no downloaded data: every input is
generated from a seed.

## Measured result

Recorded on 2026-09-22, clang 17.0.0, C++20, Release, macOS arm64, 64-bit. `steady_clock`
resolves to 41 ns and one reading costs 14 ns.

Unlike every earlier milestone, **these numbers do not reproduce exactly** — they are timings.
What reproduces is the ordering and the rough magnitude; the section on stability below says how
much they move and where that claim breaks down.

### Inference: one sample

| Benchmark | Per call | Calls/s | Cold/steady | Spread |
|---|---:|---:|---:|---:|
| MLP 784-128-10 forward | 48.5 µs | 21k | 1.0× | 0.1% |
| MLP forward, workspace reused | 48.0 µs | 21k | 1.2× | 0.0% |
| softmax regression 784-10 forward | 6.8 µs | 147k | 1.2× | 0.1% |
| CNN 8×3×3 + pool + dense forward | 68.7 µs | 15k | 1.4× | 0.8% |

The first two rows differ only in whether the forward pass allocates its scratch space or reuses
one. The difference is **0.9%, against a repetition spread of 0.1%** — real but nearly
negligible. That is worth knowing because it is the opposite of the usual advice: at 784×128
multiply-adds per call the arithmetic dominates so thoroughly that the allocator does not matter,
and the `forward(input, workspace)` overload added for the GAN milestone earns its place by
avoiding garbage, not by being faster.

The linear model is 7× faster than the MLP, which is the 100352 multiply-adds of the hidden layer
against the linear model's 7840 — a ratio of 12.8 against a measured 7.1, the rest being the
per-call overhead both pay.

### Training: one gradient over a batch of 32

| Benchmark | Per sample | Samples/s | Spread |
|---|---:|---:|---:|
| MLP gradient, batch 32 | 64.4 µs | 16k | 0.1% |
| CNN gradient, batch 32 | 112.1 µs | 9k | 0.8% |

A backward pass costs about a third more than a forward one for the MLP (64.4 against 48.5),
which is the expected shape: the backward pass does one extra matrix product per layer but reuses
the forward pass's cached activations.

### Sequence models: one 50-character window, forward and backward

| Benchmark | Per character | Characters/s | Spread |
|---|---:|---:|---:|
| elman 128, BPTT over 50 | 16.5 µs | 61k | 0.1% |
| gru 128, BPTT over 50 | 40.4 µs | 25k | 0.1% |
| lstm 128, BPTT over 50 | 52.2 µs | 19k | 0.1% |
| transformer 2×4×64, context 50 | 56.1 µs | 18k | 0.2% |

This is the cost side of milestones 15 to 17, measured directly instead of inferred from an
epoch's wall clock. The gated cells cost **2.4× (GRU) and 3.2× (LSTM)** an elman cell of the same
width — close to their gate counts of 3 and 4, with the shared output layer accounting for the
shortfall. The transformer at this size is 3.4× the elman cell.

The elman figure, 61k characters a second, matches experiment 15's 68.6k measured over whole
epochs; the difference is that this one excludes the data handling an epoch also pays for.

### Reinforcement learning

| Benchmark | Per unit | Units/s | Cold/steady | Spread |
|---|---:|---:|---:|---:|
| snake step | 37.2 ns | 27M | 19× | 4.1% |
| snake observation | 22.9 ns | 44M | 44× | 2.7% |
| tabular Q update | 1.9 ns | 525M | 88× | 0.0% |
| DQN update, batch 32 | 5.6 µs/transition | 180k | 1.0× | 0.3% |

**The tabular update is 1.9 nanoseconds** — a couple of array reads, a max over three, and a
write. One DQN update over a batch of 32 costs 178 µs, which is **93000× more** for the same
Bellman target. That is the quantitative version of experiment 18's finding that the table beat
the network on wall clock by 56×: per update the gap is enormous, and the network closes most of
it only by needing far fewer episodes.

The environment is not the bottleneck in either case: at 37 ns a step, a 150-step episode costs
5.6 µs of Snake against 26.7 ms of DQN learning — **the learning is 4800× the simulation.** Any
effort spent optimizing the environment would be wasted.

### Cold calls

The three fastest benchmarks have the largest cold ratios: the tabular update's first call is 88×
its steady state, the observation's 44×, a snake step's 19×. Nothing about the code changes
between the first call and the thousandth; what changes is that the first call takes the page
faults on a freshly allocated 48 KB table, misses in every cache, and mispredicts every branch.

This is the argument for warm-up stated as a measurement: had the harness reported a single
timing, it would have been up to 88× wrong. It is also the argument for reporting the cold figure
rather than discarding it — for something called once, 88× is the number that matters.

### Implementation size

Lines that are neither blank nor a comment, as a crude stand-in for complexity:

| Module | Header | Source | Total |
|---|---:|---:|---:|
| neural_network | 219 | 1439 | 1658 |
| transformer | 83 | 873 | 956 |
| convolution | 97 | 792 | 889 |
| rnn (three cells) | 112 | 715 | 827 |
| reinforcement | 141 | 294 | 435 |
| benchmark | 104 | 328 | 432 |
| snake | 67 | 179 | 246 |

5443 lines for the seven, every gradient in them derived by hand. The count does not parse C++ and
is only meaningful between implementations of the same algorithm — which is what it is here for,
when the NumPy and PyTorch columns exist to compare against.

Peak resident memory over the whole run was **9 MiB**, which is the entire cost of holding every
model this experiment builds.

### Stability, and the one figure that is not stable

The `spread` column measures **within-run** repeatability, and it is excellent: eleven of the
fourteen benchmarks agree across their seven repetitions to within 1%.

It says nothing about **between-run** variation, and one benchmark has a great deal of it. Across
roughly twenty-five runs of the same binary on the same machine, the CNN forward pass fell into
two clusters: about 49–69 µs, and about 137–156 µs — a factor of 2.3 — with a low spread inside
each run. Every other benchmark held to a few percent between runs; the MLP forward, measured in
the same runs, never left 42.6–48.5 µs.

Two hypotheses were tested and neither explains it. Adding an unrelated comment to the experiment
and rebuilding, three times, left it at 61–67 µs, so it is not code alignment in the experiment's
translation unit. Running it under twelve busy loops moved **every** benchmark by about 15%, not
one of them by 2.3×, so it is not scheduler or frequency behaviour either. The cause is not
established, and the figure above is reported as one measurement rather than as the CNN forward
pass's cost.

The useful lesson is the one the column already implied: a low within-run spread is necessary for
a benchmark to mean anything and nowhere near sufficient, and a single run — however carefully
warmed, repeated and median-ed — can still be a factor of two from what the same binary does an
hour later.

### Compilation

From `scripts/benchmark_build.sh`, a clean Release build at 16 parallel jobs: **0.37 s** to
configure, **1.64 s** for the library, **2.68 s** for the nineteen experiment executables and
twenty test executables. The whole project builds from nothing in under five seconds, which is
what standard-library-only and no third-party dependencies buys.

## Limitations

These are single-machine, single-compiler, single-configuration numbers: one clang, one arm64
laptop, one build type. The cross-language comparison this file's JSON format exists for does not
exist yet — the NumPy, PyTorch, JAX and Triton columns of the status table are still planned — so
nothing here is yet a comparison *between* implementations, only a measurement of one. The line
counts are a crude proxy for complexity and ignore how much of the work each line does. And the
CNN forward figure should be re-measured before it is quoted anywhere.
