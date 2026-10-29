# Triton softmax regression on MNIST

[Topic and results](../README.md) · [Mathematics](../idea.md) ·
[Triton conventions](../../../docs/implementations/triton.md)

[Kernels](kernels.py) · [Model](model.py) · [Public API](__init__.py) ·
[Experiment](experiment.py) · [Tests](tests/test_triton_softmax_regression.py)

```bash
# Linux with a CUDA GPU (Triton ships Linux wheels only)
python -m pip install -e ".[dev,numpy,triton]"
python -m ml_scratch.triton.softmax_mnist.experiment
# Equivalent installed command: ml-triton-softmax-mnist
python -m pytest topics/08_softmax_mnist/triton/tests

# Linux without a GPU: the same kernels on Triton's CPU interpreter (slow; used by CI)
TRITON_INTERPRET=1 python -m pytest topics/08_softmax_mnist/triton/tests
```

A training step is two fused float64 [kernels](kernels.py):

1. **`softmax_rows`** runs one program per 16 batch rows. It loads each row through the batch's
   index array, so gathering the shuffled batch is part of the load and no batch is copied. It
   scales the row's `uint8` pixels by 1/255 in registers, so no float copy of MNIST exists. The
   logits `X W^T + b` accumulate over 32-feature tiles as a broadcast multiply and sum. The ten
   classes are padded to a 16-wide tile, and padding classes get `-inf`. The same pass applies the
   max-shifted softmax and writes three outputs: `p - y`, the per-row loss
   `log-sum-exp(z) - z_y`, and the first-maximum argmax.
2. **`reduce_and_update`** gives each program one (class tile, 32-feature tile) of `W`. The
   program sums `(p - y)^T x` over the batch for its tile and immediately applies
   `W -= lr (G / B + l2 W)`. One program per class tile also updates the biases. The weight
   gradient therefore exists only in registers. Each weight has exactly one owner, so the
   reduction needs no atomics. Its order is fixed, so training is bit-reproducible: a test
   requires two runs to produce identical parameters.

`gradient()` launches the same pair with the update switched off (`APPLY=False`). Evaluation,
which covers loss, predictions, and the confusion matrix, uses only `softmax_rows`, in 8192-row
launches. The host model mirrors the other backends' API: parameters live in one flat float64
device tensor with weight and bias views. The scalar shuffle stream's order is uploaded once per
epoch.

Two precision details matter, and tests pin both:
- **Scalar kernel arguments.** Triton passes a Python `float` as fp32. The first version passed
  `1 / B`, the learning rate, and the L2 coefficient that way, and its parameters drifted from
  NumPy's by 1e-9 after three epochs. Now `1 / B` is computed in-kernel from the integer size, and
  the hyperparameters arrive as a two-element float64 device array. The trajectories agree with
  the scalar model within `rtol=1e-13`.
- **Compile-time loop bounds.** The feature count `D` is a `constexpr`. The batch loop runs to
  `MAX_B`, the batch size rounded up to a power of two, and the true size is used only in masks.
  Full and partial training batches share one compiled kernel. This also lets the kernels run
  under `TRITON_INTERPRET=1`. The interpreter wraps runtime integers in one-element NumPy arrays,
  which current NumPy refuses to turn into a Python loop bound.

The [experiment](experiment.py) is the scalar protocol with this backend. On an RTX 5080 its
output equalled the scalar output exactly, apart from wall-clock time; see the
[topic results](../README.md#python-numpy-pytorch-and-triton-reproduction). This model is float64
and launches two small kernels per 64-image batch, so its timing says nothing about what a
float32, larger-batch GPU kernel could do. No speedup is claimed.

[Tests](tests/test_triton_softmax_regression.py) skip unless Triton is installed and either a
CUDA device or `TRITON_INTERPRET=1` is present. They cover:
- trajectories, gradients, predictions, probabilities, and confusion matrices against the scalar
  model, for 4, 10, and 17 classes (17 pads to a 32-wide tile) and odd feature counts and batch
  sizes;
- equality of the byte and float paths, bit-reproducibility, and chunked evaluation;
- the float64 scalar-argument regression, saturation, divergence, and invalid input;
- with MNIST, the protocol subset and the full experiment's recorded results.

All tests pass on the GPU and, apart from the full experiment, under the interpreter.
