# Repository layout

[Design guides](README.md) · [Implementations](../../src/README.md)

```text
.
├── .github/           # Continuous integration and pull-request template
├── docs/              # Setup, status, roadmap, findings, mathematics, and design
├── experiments/       # Numbered topic READMEs and runnable experiment drivers
├── results/           # Measurement policy and ignored generated output directories
├── scripts/           # Dataset downloads, build measurement, and documentation checks
├── src/
│   ├── cpp/           # C++ interfaces, core algorithms, utilities, and private declarations
│   ├── python/        # Installable ml_scratch package and backend guides
│   └── triton/        # Planned GPU-kernel implementation guide
├── tests/
│   ├── cpp/           # CTest executables and small fixtures
│   ├── python/        # Pure-Python model and experiment tests
│   └── tooling/       # Repository-tool tests
├── AGENTS.md          # Agent/contributor requirements
├── CMakeLists.txt     # C++ library, experiment, and test targets
├── CONTRIBUTING.md    # Contributor workflow
├── LICENSE
├── pyproject.toml     # Python package, optional dependencies, CLIs, and quality tools
└── README.md          # Concise project entry point and topic links
```

## Ownership

| Kind of content | Location | Rule |
|---|---|---|
| Model algorithm | `src/<implementation>/` | Forward passes, objectives, gradients, and updates stay explicit |
| Experiment | `experiments/<number>_<topic>/` | Own datasets, splits, selection, configuration, and reporting |
| Topic guide | Experiment `README.md` | Link the derivation and implementation; document the runnable experiment |
| Mathematical derivation | `docs/mathematics/` | Explain equations, numerical conventions, and limitations |
| Implementation guide | Implementation `README.md` | Map source modules and describe setup and boundaries |
| Tests | `tests/<implementation>/` | Test independent expectations, edge cases, and reproducibility |
| Tooling | `scripts/`, `tests/tooling/` | Keep development helpers separate from model packages |
| Measured output | `results/` | Record command, configuration, seed, environment, and method |
| Downloaded/generated files | Ignored local directories | Keep datasets, models, environments, and build products out of Git |

Topic numbers are stable across implementations. Reuse an existing topic's model when a later
experiment extends it; shared source is linked from each relevant README. A planned implementation
has an explicit status page instead of placeholder model classes.

## Local directories

`build/`, `build-*/`, `cmake-build-*/`, `.venv/`, caches, IDE state, `data/`, and generated artifacts
are ignored. They may exist in a working checkout without being part of the published repository.
The layout above describes maintained source and documentation, not local execution state.

The full milestone matrix lives in [status](../status.md). Keep the root README's implementation
summary current, and place detailed findings in topic READMEs or the [findings guide](../findings.md).
