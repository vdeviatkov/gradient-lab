# Topic ownership and repository layout

[Design guides](README.md#architecture) · [Canonical topic/status matrix](../README.md#topics)

```text
topics/<number>_<topic>/
  README.md                     # Experiment method, configuration, results, limitations
  idea.md                       # Mathematical explanation and design decisions
  cpp/
    README.md                   # Source map and C++ implementation notes
    CMakeLists.txt              # Topic source, executable, and test registration
    main.cpp                    # Experiment driver
    <model>.cpp                 # Algorithm, if this topic owns one
    include/ml_scratch/         # Public interfaces, if owned here
    utils/                      # Topic-specific evaluation, persistence, diagnostics
    tests/                      # Topic-specific CTest executables
  python/                       # Present when the Python implementation exists
    README.md
    __init__.py                 # Public API exports only
    model.py                    # Canonical algorithm
    experiment.py               # Experiment orchestration and installed CLI
    tests/
shared/                         # Common infrastructure used by multiple topics
docs/                           # Setup, backend conventions, design, contribution guidance
tests/                          # Cross-topic integration and repository-tool checks
scripts/                        # Data downloads, measurement tooling, documentation checks
results/                        # Generated-output policy; measurements remain ignored
```

## One owner per piece of code or information

- A topic's `idea.md` owns its mathematics. Link it from every backend instead of copying it.
- A topic's `README.md` owns its experiment report and measured results. There is no separate
  findings document or parallel experiment folder.
- An implementation lives in that topic's backend folder, including its tests and driver.
- A later topic reusing a model links its owning topic. For example, the MNIST MLP and
  regularization topics use the dense network in topic 07; LSTM/GRU uses the recurrent model in
  topic 15. No model copy is created just to fill a folder.
- Truly common operations live under `shared/`; do not move a model there merely because a later
  topic calls it. The [shared README](../shared/README.md) defines the infrastructure boundary.
- The root topic table is the single catalog and implementation-status matrix. Language guides
  describe conventions rather than repeating the catalog.
- Planned backends gain a local directory with their implementation, not a collection of empty
  placeholder code or copied READMEs.

The topic sequence is stable. Each C++ folder registers itself with the root build, and Python's
explicit package map installs topic-local packages without mirroring source into another tree.
Existing model include names, Python model imports, installed CLI names, and `build/cpp_*`
executables are retained. The n-gram baseline now has its own `ml_scratch/char_ngram.hpp` header
in topic 15 rather than being declared in the shared corpus header. Experiment module paths now use `ml_scratch.pure.<model>.experiment`.

## Local artifacts

`build/`, `build-*/`, `cmake-build-*/`, `.venv/`, caches, IDE state, `data/`, and generated output
remain local and ignored. They are not part of the maintained source layout. The package/build
configuration remains at the root, with [contributor guidance](../CONTRIBUTING.md) and the license.
