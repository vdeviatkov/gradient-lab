# Measuring implementation performance

[Topic and results](README.md) · [C++ implementation](cpp/README.md)

A benchmark measures a defined unit of work under stated conditions. Keep setup and reset work
outside the timed region, make computed values observable to the optimizer, and distinguish the
first cold call from warmed steady-state execution.

If one timed repetition performs N iterations in elapsed time T, its per-iteration estimate is
T / N. Repeat the measurement and retain each repetition; summarize with the median rather than
presenting a single run as a stable property. For W items of work per iteration, throughput is
W divided by the median per-iteration time. Record units and the operation represented by W.

Measure clock resolution and overhead before interpreting short timings. Auto-tune the iteration
count so elapsed time rises above that floor. Record compiler, build type, hardware, memory, and
measurement configuration. Build time is a separate measurement because a running executable
cannot time its own compilation.

Different batches, update counts, preprocessing, and initialization can invalidate an apparent
comparison. Match behavior first. Report variability, and treat noisy or non-repeatable timings as
limitations rather than manufacturing a clean ranking. Source-line counts are a rough accounting
of implementation size, not a measure of algorithmic quality.

The [topic report](README.md) is the single home for this experiment's configuration, measured
results, acceptance criteria, and known limitations.
