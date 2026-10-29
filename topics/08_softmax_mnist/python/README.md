# Pure-Python softmax regression on MNIST

[Topic and measured results](../README.md) · [Mathematics](../idea.md) ·
[Python conventions](../../../docs/implementations/pure-python.md)

[Model](model.py) · [Metrics](metrics.py) · [Public API](__init__.py) ·
[Experiment](experiment.py) · [Tests](tests/test_softmax_regression.py)

```bash
python -m pip install -e ".[dev]"
./scripts/download_mnist.sh                         # data/ is not committed
python -m ml_scratch.pure.softmax_mnist.experiment  # about four minutes
# Equivalent installed command: ml-softmax-mnist
python -m pytest topics/08_softmax_mnist/python/tests
```

The standard-library [model](model.py) keeps one weight row and one bias per class. It computes
`z_k = w_k . x + b_k` and `p = softmax(z)`, and fits them by mini-batch SGD on the mean
cross-entropy plus an optional `(l2 / 2) ||W||^2`. The penalty covers the weights only, never the
biases. The output delta is `p - y`, so a batch's gradient is `sum (p - y) x^T / B`. The
parameters start at zero, which is safe because the loss is convex. Every `exp` and `log` is
evaluated in max-shifted form.

```python
from ml_scratch.pure.softmax_mnist import SoftmaxRegression, SoftmaxTrainingConfig

model = SoftmaxRegression(feature_count=2, class_count=3)
data = [([1.0, 0.0], 0), ([0.0, 1.0], 1), ([-1.0, -1.0], 2)]
result = model.fit(data, SoftmaxTrainingConfig(learning_rate=0.5, max_epochs=200))
assert model.accuracy(data) == 1 and result.loss_per_epoch[-1] < result.loss_per_epoch[0]
```

**Sparsity is what makes standard-library MNIST practical.** About 81% of MNIST pixels are zero
(150 nonzero of 784 on average). A zero feature adds `w * 0 = 0` to every score and
`(p - y) * 0 = 0` to every weight gradient, so `SparseDataset` stores only the nonzero
`(indices, values)` of each row. Scores gather those weights with `operator.itemgetter` and sum
the products in C. Without a penalty, an SGD step updates only the weights whose features appeared
in the batch. With L2, every weight decays every step, so the whole matrix is updated. The result
is exact, not an approximation: a test checks that sparse and dense rows give the same scores and
loss. One epoch over 54000 images takes about 5.8 s.

[Metrics](metrics.py) computes accuracy and per-class and macro precision, recall, and F1 from
`confusion[actual][predicted]`. A zero denominator is defined as zero, so a class the model never
predicts shows up explicitly.

The [experiment](experiment.py) reads MNIST through the shared, standard-library
[IDX reader](../../../shared/python/ml_scratch/pure/mnist.py). The reader applies the same header
checks and the same `ML_SCRATCH_MNIST_DIR` override as the C++ loader. Each pixel byte becomes
`byte / 255.0`, the C++ loader's only preprocessing step. The protocol is the C++ one:
- the last 6000 training images validate;
- six learning-rate and L2 pairs are trained for 8 epochs and ranked by validation accuracy, with
  the first best kept;
- the winner is retrained for 30 epochs;
- the official test file is evaluated once.

`run_with` accepts any backend's model class and data-preparation function, so the NumPy,
PyTorch, and Triton experiments run this protocol unchanged.

Mini-batch shuffling uses a local `random.Random(seed)`. The C++ `std::shuffle` is not specified
bit-for-bit across standard libraries, so training follows a different trajectory from C++. The
[recorded results](../README.md#python-numpy-pytorch-and-triton-reproduction) show what that changes.

[Model tests](tests/test_softmax_regression.py) cover:
- the zero-model loss and gradient by hand, the parameter layout, and central differences with and
  without L2;
- sparse-versus-dense exactness, learning, reproducibility, and `target_loss`;
- penalty-only decay of untouched weights, saturated logits, and the confusion matrix and metrics;
- divergence and invalid input.

[Experiment tests](tests/test_softmax_mnist_experiment.py) run the whole protocol on a generated
IDX directory and, when MNIST is present, on a 3000-image subset. The shared reader has its own
[tests](../../../tests/python/test_mnist.py).
