# XOR multilayer perceptron — pure Python

[Topic and results](../README.md) · [Idea](../idea.md) · [Python setup](../../../docs/implementations/pure-python.md)

The canonical algorithm lives in [model.py](model.py). The [experiment](experiment.py) owns
the dataset/configuration choices and report. [__init__.py](__init__.py) exports the public model
API without duplicating its implementation.

## Run

After installing the repository from its root:

```bash
python -m ml_scratch.pure.mlp.experiment
```

The installed CLI name remains `ml-xor-mlp`.

## Tests

- [test_xor_mlp.py](tests/test_xor_mlp.py)
- [test_xor_mlp_experiment.py](tests/test_xor_mlp_experiment.py)

Run `python -m pytest topics/02_xor_mlp/python/tests` from the repository root.
