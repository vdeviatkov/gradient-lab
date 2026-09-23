# Benchmarks

Measurements produced by `./build/cpp_benchmarks` and `./scripts/benchmark_build.sh`, with the
environment metadata needed to reproduce them. See
[experiment 19](../../experiments/19_benchmarks/README.md) for the methodology and the reading.

- `cpp.json` — run-time measurements: every repetition of every benchmark, plus the compiler,
  platform, build type, clock characteristics and peak resident memory of the run that produced
  them. Both files are ignored by Git, like everything else under `results/`; run the experiment
  to regenerate them.
- `cpp_build.json` — configure and compile times for a clean Release build.

These are timings, so unlike the rest of the repository they do not reproduce exactly. The
experiment refuses to publish a run whose own repetitions disagree by more than 20%, which is
what happens when the machine is not idle, and experiment 19 documents the one benchmark whose
between-run variation is large enough that it should be re-measured before being quoted.

When the NumPy, PyTorch, JAX and Triton implementations exist, their results belong here in the
same shape, with `"implementation"` naming each one.
