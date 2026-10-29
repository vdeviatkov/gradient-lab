# PyTorch softmax regression on MNIST

[Topic and results](../README.md) · [Mathematics](../idea.md) ·
[PyTorch conventions](../../../docs/implementations/pytorch.md)

[Model](model.py) · [Public API](__init__.py) · [Experiment](experiment.py) ·
[Tests](tests/test_torch_softmax_regression.py)

```bash
python -m pip install -e ".[dev,pytorch]"
python -m ml_scratch.pytorch.softmax_mnist.experiment
# Equivalent installed command: ml-pytorch-softmax-mnist
python -m pytest topics/08_softmax_mnist/pytorch/tests
```

This is the [NumPy design](../numpy/README.md) in CPU float64 tensors:
- batched scores `X W^T + b` and an in-place `P - Y` delta;
- the hand-derived gradients `(P - Y)^T X` and the column sums of `P - Y`;
- SGD with weight decay under `torch.no_grad()`;
- byte images scaled one batch at a time, and chunked full-split evaluation.

No `nn.Linear`, loss module, or optimizer is involved.

`autograd_gradient` is an independent oracle. It clones the flat parameter tensor as a leaf that
requires gradients and rebuilds the weight and bias views over that leaf. It then evaluates the
max-shifted cross-entropy plus the L2 penalty and calls `torch.autograd.grad`. It never calls the
manual gradient, and it never mutates the model or populates `.grad`. Tests require the two to
agree within `rtol=1e-12`, with and without L2.

The configuration, result types, and shuffle stream are the scalar ones, so a run follows the
scalar and NumPy trajectories. On full MNIST it printed the scalar output exactly, apart from
wall-clock time. `predict` and `accuracy` return Python numbers. Probabilities, predictions,
gradients, parameters, and confusion matrices return tensors.

[Tests](tests/test_torch_softmax_regression.py) mirror the NumPy suite and add the autograd oracle.
Only CPU behavior is tested.
