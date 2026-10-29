# PyTorch backpropagation and gradient checking

[Topic and results](../README.md) · [Mathematics](../idea.md) ·
[PyTorch conventions](../../../docs/implementations/pytorch.md)

[Model](model.py) · [Gradient check](gradient_check.py) · [Public API](__init__.py) ·
[Experiment](experiment.py) · [Tests](tests/test_torch_backpropagation.py)

```bash
python -m pip install -e ".[dev,pytorch]"
python -m ml_scratch.pytorch.backpropagation.experiment
# Equivalent installed command: ml-pytorch-backpropagation
python -m pytest topics/07_backpropagation/pytorch/tests
```

This is the [NumPy design](../numpy/README.md) in CPU float64 tensors. A batched forward pass
computes `Z = A W^T + b`. The backward pass applies the hand-derived products `Delta^T A_prev`,
the row sum of `Delta`, and `(Delta W) * g'(Z_prev)`. Each layer's `W` and `b` are views into
one flat parameter tensor, and `__deepcopy__` rebuilds those views for gradient-check probes.
Training, the manual gradient, and the finite differences run under `torch.no_grad()`. No
`nn.Linear`, `nn.Module`, loss module, or optimizer is used.

`autograd_gradient` is an independent oracle. It clones the flat tensor as a leaf that requires
gradients, rebuilds the views over that leaf, and runs the same forward and loss code inside
`torch.enable_grad()`. It then calls `torch.autograd.grad`. It never calls the manual backward
pass, and it never mutates the model's parameters or populates `.grad`. Tests require the manual
gradient to equal autograd within `rtol=1e-12`, and to match central differences, for every
configuration in the experiment.

Inputs are converted to detached CPU float64 at the boundary, so a feature tensor that requires
gradients does not drag its graph into the model. `predict_class` and `accuracy` return Python
numbers. Gradients, parameters, forward outputs, and predictions return tensors. The value types,
initialization, and shuffle stream are the scalar model's, so a run follows the scalar and NumPy
trajectories up to floating-point reduction order. The [experiment](experiment.py) is the scalar
protocol with this backend's network and checker.

[Tests](tests/test_torch_backpropagation.py) mirror the NumPy suite: scalar agreement, matching
training trajectories, view aliasing, the checker and error orders, stability, invalid input, and
the full experiment. They add the autograd oracle and the detachment of tensor inputs that require
gradients. Only CPU behavior is tested.
