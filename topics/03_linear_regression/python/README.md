# Linear regression — pure Python

[Topic and results](../README.md) · [Idea](../idea.md) · [Python setup](../../../docs/implementations/pure-python.md)

The canonical algorithm lives in [model.py](model.py). The [experiment](experiment.py) owns
the dataset/configuration choices and report. [__init__.py](__init__.py) exports the public model
API without duplicating its implementation.

## Run

After installing the repository from its root:

```bash
python -m ml_scratch.pure.linear_regression.experiment
```

The installed CLI name remains `ml-linear-regression`.

## Tests

- [test_linear_regression.py](tests/test_linear_regression.py)
- [test_linear_regression_experiment.py](tests/test_linear_regression_experiment.py)

Run `python -m pytest topics/03_linear_regression/python/tests` from the repository root.
