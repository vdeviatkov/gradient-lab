# Tests

[Quality-check commands](../docs/testing.md) · [Topic owners](../README.md#topics)

Model and experiment tests live with their implementations under
`topics/<topic>/<backend>/tests/`. Each implementation README links those tests.
This directory contains only shared integration and repository-tool tests.

| Scope | Command / guide |
|---|---|
| All Python and tooling tests | `python -m pytest` after editable installation |
| Topic Python tests | `python -m pytest topics/<topic>/python/tests` |
| All C++ topic/integration tests | `ctest --test-dir build --output-on-failure` |
| Shared C++ integration | [Guide](cpp/README.md) |
| Shared Python fixtures | [Guide](python/README.md) |
| Repository tooling | [Guide](tooling/README.md) |

Tests use small deterministic fixtures and need no external datasets or GPU. Keep numerical
expectations independent of the implementation. Performance claims belong to reproducible topic
experiments, not portable CI expectations.
