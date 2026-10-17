# Tests

[Quality-check commands](../docs/testing.md) · [Contributor guide](../CONTRIBUTING.md)

| Suite | Runner | Guide |
|---|---|---|
| Pure-Python models and experiments | `python -m pytest tests/python` | [README](python/README.md) |
| C++ algorithms and support utilities | `ctest --test-dir build --output-on-failure` | [README](cpp/README.md) |
| Repository tooling | `python -m pytest tests/tooling` | [README](tooling/README.md) |

Tests use small deterministic fixtures and need no downloaded datasets or accelerator hardware.
Keep numerical checks independent of the implementation under test. Experimental performance
measurements belong in the experiment READMEs, not hard-coded as general-purpose test claims.
