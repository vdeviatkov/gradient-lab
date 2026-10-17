# Results policy

[Topic reports](../README.md#topics) · [Reproducibility](../docs/reproducibility.md)

Only measured, reproducible outputs belong here. Keep human-readable methods, configuration,
results, interpretation, and limitations in the owning topic's README; do not maintain a second
report here.

Every published measurement records the exact command, implementation/version, dataset identity,
preprocessing, splits, configuration, seeds, software/hardware environment, and measurement method.
For performance work, state build type, timing boundaries, warm-up, repetitions, and variability.
Never substitute estimates for experimental results.

## Benchmarks

Generated files under `results/benchmarks/` are ignored:

- `cpp.json`: runtime repetitions and compiler, platform, build, clock, and memory metadata.
- `cpp_build.json`: configure and compilation times for a clean Release build.

Regenerate them with `./build/cpp_benchmarks` and `./scripts/benchmark_build.sh`. See
[topic 19](../topics/19_benchmarks/README.md) for methodology, acceptance criteria, results, and
limitations. Future implementations use the same report shape with their own implementation name.
Timings do not reproduce exactly, even when model behavior does.

## Figures

Generated plots and images under `results/figures/` are ignored. Curate and explicitly add only
figures backed by a documented experiment. Scripts create output directories when needed; empty
placeholder folders are not required.

Large datasets, trained models, local environments, and build products stay out of Git.
