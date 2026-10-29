# NumPy backpropagation and gradient checking

[Topic and results](../README.md) · [Mathematics](../idea.md) ·
[NumPy conventions](../../../docs/implementations/numpy.md)

[Model](model.py) · [Gradient check](gradient_check.py) · [Public API](__init__.py) ·
[Experiment](experiment.py) · [Tests](tests/test_numpy_backpropagation.py)

```bash
python -m pip install -e ".[dev,numpy]"
python -m ml_scratch.numpy.backpropagation.experiment
# Equivalent installed command: ml-numpy-backpropagation
python -m pytest topics/07_backpropagation/numpy/tests
```

This is the scalar network with the sample loop replaced by matrix products. Samples are rows,
so a layer computes `Z = A W^T + b` for the whole batch. The backward pass handles every sample
at once with three hand-derived products:

```text
dL/dW      = Delta^T A_prev        (the sum of per-sample outer products delta a^T)
dL/db      = Delta summed over rows
Delta_prev = (Delta W) * g'(Z_prev)
```

`Delta` already carries the `1/n` of the mean loss. No autodiff, optimizer, or neural-network
library is involved.

All parameters live in one flat float64 vector, in the scalar layout. Each layer's `W` and `b`
are reshaped views into it, so the vector that the gradient check perturbs is the same storage
the forward pass reads. `set_parameters` copies into that storage in place, and `parameters()`
returns a copy. A generic `deepcopy` would turn the views into independent arrays and silently
detach them, so `__deepcopy__` rebuilds the views over the copy's own storage. A test covers this.

Central differences perturb one coordinate of a copy's flat vector at a time, and every loss
evaluation is a single batched pass. The comparison is vectorized: norms, a first-maximum
`argmax` for the worst entry, and the floored elementwise ratio.

`Activation`, `Loss`, `DenseLayer`, the training config and result, and `GradientCheckResult` are
the scalar value types. Initialization reuses the scalar C++-compatible draws, and the shuffle
uses the same local `random.Random(seed)`. A NumPy run therefore follows the scalar trajectory up
to floating-point reduction order. The [experiment](experiment.py) runs the scalar protocol
unchanged with this backend's network and checker.

Overflow in a diverging run is reported as `ArithmeticError` by the epoch-loss check, without
NumPy runtime warnings. Saturated logits use `logaddexp` and max-shifted log-sum-exp.

[Tests](tests/test_numpy_backpropagation.py) cover:
- agreement with the scalar network on every experiment configuration: parameters, loss,
  gradient, batched and single forward passes, predictions, and accuracy;
- 40-epoch mini-batch loss histories and final parameters equal to scalar Python within
  `rtol=1e-11`;
- view aliasing, the checker against scalar `check_gradient`, and error-order ratios;
- stable losses, divergence, and invalid input;
- the full experiment, including the recorded scalar training results.
