# Pure-Python backpropagation and gradient checking

[Topic and measured results](../README.md) · [Mathematics](../idea.md) ·
[Python conventions](../../../docs/implementations/pure-python.md)

[Model](model.py) · [Gradient check](gradient_check.py) · [Public API](__init__.py) ·
[Experiment](experiment.py) · [Tests](tests/test_backpropagation.py)

```bash
python -m pip install -e ".[dev]"
python -m ml_scratch.pure.backpropagation.experiment   # about two minutes
# Equivalent installed command: ml-backpropagation
python -m pytest topics/07_backpropagation/python/tests
```

The standard-library [model](model.py) is a general stack of dense layers,
`z = W a + b`, `a' = g(z)`, with identity, sigmoid, tanh, and ReLU activations. It supports
three losses: mean squared error, binary cross-entropy, and softmax cross-entropy. The two
cross-entropy losses apply their squashing inside the loss, so the final layer must be linear and
the output delta collapses to `p - y`. Backpropagation is one forward sweep that caches every `z`
and `a`, then one backward sweep that applies `dL/dW = delta a^T`, `dL/db = delta`, and
`delta_prev = (W^T delta) * g'(z_prev)`. Training is plain mini-batch gradient descent.

```python
from ml_scratch.pure.backpropagation import (
    Activation,
    DenseLayer,
    FeedForwardNetwork,
    Loss,
    check_network_gradient,
)

layers = (DenseLayer(2, 4, Activation.TANH), DenseLayer(4, 3))
network = FeedForwardNetwork(layers, Loss.SOFTMAX_CROSS_ENTROPY, seed=7)
data = [((0.2, -0.4), (1.0, 0.0, 0.0)), ((1.0, 0.3), (0.0, 0.0, 1.0))]
assert check_network_gradient(network, data).passed
```

Parameters are also exposed as one flat vector in the C++ layout: layer by layer, the weight
matrix row by row, then the biases. That flat vector is what makes the [gradient
check](gradient_check.py) general, because any architecture becomes a function `R^P -> R`. The
check perturbs a deep copy with central differences and compares the result using the norm ratio
`||a - n|| / (||a|| + ||n||)`. It reports the largest absolute and floored elementwise
disagreements as diagnostics. `forward_difference_gradient` exists only to show why the central
form is used.

Initialization reproduces the C++ network word for word. C++ seeds `std::mt19937` with
`init_genrand`, while Python's `random.seed` uses `init_by_array`. `mt19937(seed)` therefore
builds the C++ state and installs it with `random.Random.setstate`, so `getrandbits(32)` returns
the same words. Weights are Glorot or He uniform drawn from those words, and biases start at
zero. A test checks the C++ standard's required 10000th output, and the experiment matches the
recorded C++ values wherever they depend only on the parameters. Mini-batch shuffling uses a
local `random.Random(seed)`, because `std::shuffle` is not specified bit-for-bit across C++
libraries. Training trajectories therefore differ from C++.

Gradient-check ratios at `epsilon = 1e-5` are as small as C++'s or smaller. On the
270-sample spiral checks they are 4–8 times smaller; on the 3- and 4-sample checks they differ by
at most about 2 times. Since CPython 3.12, the built-in `sum()` uses compensated (Neumaier)
summation for floats, so each loss over many samples carries less round-off. Replacing it with a naive left-to-right sum puts the softmax ratio
at `4.102e-10`, against C++'s `4.052e-10`. The round-off floor of a gradient check is a property
of how accurately the loss is computed, not only of `epsilon`.

[Tests](tests/test_backpropagation.py) cover:
- the MT19937 reference output, initialization limits, and the parameter layout;
- forward passes, losses, and gradients computed by hand, plus saturated logits;
- central-difference agreement for every loss and activation, soft targets, and random
  architectures;
- the second-order versus first-order error decay;
- detection of a corrupted entry and the near-zero elementwise caveat;
- XOR and three-class training, reproducibility, divergence, and invalid input.

[Experiment tests](tests/test_backpropagation_experiment.py) check the interleaved spiral splits
and the C++-matching initialization. They also run a 100-epoch version of the experiment that
must meet every criterion. The full 4000-epoch run is recorded in the
[topic README](../README.md#python-numpy-and-pytorch-reproduction).
