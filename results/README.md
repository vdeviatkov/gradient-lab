# Results

[Experiments](../README.md#topics) · [Topic reports](../README.md#topics)

Only measured, reproducible outputs belong here. Never publish estimates as experimental results.
For every report, record:

- The exact command and implementation/version.
- Dataset identity, preprocessing, splits, configuration, and deterministic seeds.
- Software and hardware environment.
- Measurement method, including build type, warm-up, repetitions, and timing boundaries where relevant.
- Evaluation criteria, observed outcomes, and limitations.

| Output | Location | Policy |
|---|---|---|
| Runtime and build measurements | [benchmarks/](benchmarks/README.md) | Generated files are ignored; retain reproduction instructions |
| Generated plots and images | [figures/](figures/README.md) | Ignored by default; curate only figures backed by a documented experiment |
| Human-readable methods and findings | [Topic READMEs](../README.md#topics) | Keep configuration, evidence, and interpretation together |

Large datasets, trained model files, local environments, and build outputs do not belong in Git.
A seed guarantees reproducibility only within the stated generator/toolchain contract; see
[reproducibility](../docs/design/reproducibility.md).
