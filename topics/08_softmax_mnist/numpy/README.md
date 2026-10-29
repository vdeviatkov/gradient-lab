# NumPy softmax regression on MNIST

[Topic and results](../README.md) · [Mathematics](../idea.md) ·
[NumPy conventions](../../../docs/implementations/numpy.md)

[Model](model.py) · [Public API](__init__.py) · [Experiment](experiment.py) ·
[Tests](tests/test_numpy_softmax_regression.py)

```bash
python -m pip install -e ".[dev,numpy]"
python -m ml_scratch.numpy.softmax_mnist.experiment
# Equivalent installed command: ml-numpy-softmax-mnist
python -m pytest topics/08_softmax_mnist/numpy/tests
```

Samples are rows, so one batch is a matrix `X` of shape `(B, 784)`. The model works in three
steps:
- **Scores:** `Z = X W^T + b`.
- **Output delta:** `P - Y`, written in place by subtracting 1 at each row's label.
- **Gradients:** the hand-derived products `G_W = (P - Y)^T X` and `g_b`, the column sums of
  `P - Y`.

The SGD step applies `W -= lr (G_W / B + l2 W)` and `b -= lr g_b / B`. The softmax and
log-sum-exp are max-shifted per row. One flat float64 vector holds the parameters in the scalar
layout, and `weights` and `biases` are views into it. `confusion_matrix` counts every
(actual, predicted) pair with a single `bincount`.

`ByteImageDataset` keeps the images as the stored `uint8` bytes and converts only each gathered
batch, as `byte / 255.0`. That yields exactly the scalar model's pixel values, and resident image
memory is 42 MB instead of a 339 MB float64 copy. Full-split loss, predictions, and gradients run
in 4096-row chunks for the same reason. Plain `(features, label)` lists are also accepted.

The configuration and result types are the scalar ones, and the shuffle is the same local
`random.Random(seed)`. A NumPy run therefore follows the scalar trajectory up to floating-point
reduction order. On full MNIST it printed the scalar output exactly, apart from wall-clock time.
The [experiment](experiment.py) is the scalar protocol with this model and dataset.

[Tests](tests/test_numpy_softmax_regression.py) cover:
- trajectories, parameters, gradients, predictions, and confusion matrices that match the scalar
  model, with and without L2, on odd-sized byte images;
- chunked evaluation equal to a single pass, exact byte scaling, saturation, divergence, and
  invalid input;
- when MNIST is present, the protocol on a 2000-image subset against scalar Python, and the full
  experiment with its recorded results.
