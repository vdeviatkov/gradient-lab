# Perceptron and logic gates — pure Python

[Topic and results](../README.md) · [Idea](../idea.md) · [Python setup](../../../docs/implementations/pure-python.md)

The canonical algorithm lives in [model.py](model.py). The [experiment](experiment.py) owns
the dataset/configuration choices and report. [__init__.py](__init__.py) exports the public model
API without duplicating its implementation.

## Run

After installing the repository from its root:

```bash
python -m ml_scratch.pure.perceptron.experiment
```

The installed CLI name remains `ml-perceptron-logic-gates`.

## Tests

- [test_perceptron.py](tests/test_perceptron.py)
- [test_perceptron_experiment.py](tests/test_perceptron_experiment.py)

Run `python -m pytest topics/01_perceptron_logic_gates/python/tests` from the repository root.
