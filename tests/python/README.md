# Shared Python tests

[All tests](../README.md) · [Python conventions](../../docs/implementations/pure-python.md)

[Dataset tests](test_datasets.py) cover the shared immutable logic-gate fixtures.
[Numerical primitive tests](test_numerics.py) cover the common sigmoid and softplus functions. Model and
experiment tests live in their owning topic's `python/tests/` directory, linked from the
[root topic table](../../README.md#topics).

Run the complete suite with `python -m pytest` after an editable installation. Running only
`python -m pytest tests/python` checks common fixtures, not all topic models.
