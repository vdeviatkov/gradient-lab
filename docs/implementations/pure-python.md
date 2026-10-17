# Pure-Python conventions

[Implementation guides](README.md) · [Setup](../getting-started.md) · [Topics](../../README.md#topics)

Requires Python 3.11+. Models have no runtime dependencies. Install the checkout before running
experiments or tests:

```bash
python -m pip install -e ".[dev]"
python -m ml_scratch.pure.logistic_regression.experiment
```

Each implemented `topics/<topic>/python/` owns `model.py`, `experiment.py`, `tests/`, and a README.
Its `__init__.py` explicitly re-exports the model's public API; it does not repeat model logic.
For example, `ml_scratch.pure.logistic_regression` is installed from topic 04's Python directory.
The package mapping is explicit in [pyproject.toml](../../pyproject.toml) and supports editable
installs and ordinary wheels. Do not copy files into a second package tree.

Shared data fixtures and stable scalar functions live in
[shared/python/ml_scratch/pure/](../../shared/python/ml_scratch/pure). Model imports retain the
`ml_scratch.pure.<model>` API. Experiment modules now sit under their model package as
`ml_scratch.pure.<model>.experiment`; installed `ml-*` command names are unchanged.

Use hand-derived gradients, local seeded generators, and explicit numerical conventions.
Compare against independent central differences and keep non-convergence visible. See the topic
READMEs for commands, tests, and comparisons with C++.
