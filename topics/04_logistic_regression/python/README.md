# Binary logistic regression — pure Python

[Topic and results](../README.md) · [Idea](../idea.md) · [Python setup](../../../docs/implementations/pure-python.md)

The canonical algorithm lives in [model.py](model.py). The [experiment](experiment.py) owns
the dataset/configuration choices and report. [__init__.py](__init__.py) exports the public model
API without duplicating its implementation.

## Run

After installing the repository from its root:

```bash
python -m ml_scratch.pure.logistic_regression.experiment
```

The installed CLI name remains `ml-logistic-regression`.

## Tests

- [test_logistic_regression.py](tests/test_logistic_regression.py)
- [test_logistic_regression_experiment.py](tests/test_logistic_regression_experiment.py)

Run `python -m pytest topics/04_logistic_regression/python/tests` from the repository root.
