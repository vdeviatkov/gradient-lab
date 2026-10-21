# PyTorch logistic regression

[Topic and results](../README.md) · [Mathematics](../idea.md) ·
[PyTorch conventions](../../../docs/implementations/pytorch.md)

[Model](model.py) · [Public API](__init__.py) · [Experiment](experiment.py) ·
[Tests](tests/test_torch_logistic_regression.py)

```bash
python -m pip install -e ".[dev,pytorch]"
python -m ml_scratch.pytorch.logistic_regression.experiment
python -m pytest topics/04_logistic_regression/pytorch/tests
```

All numerical model tensors use CPU float64. Inputs accept the existing pair datasets or matrices;
external tensors are detached and normalized at the boundary. See the shared conventions for
return types, configuration reuse, and numerical-comparison limits.

The tensor model uses affine logits and sample-mean weighted BCE. Positive examples receive
weight alpha; normalization is by sample count rather than total class weight. The manual logit
derivative is `(sigmoid(z) - y) * where(y == 1, alpha, 1)`; matrix reductions give the parameter
gradients. Target-specific `logaddexp` preserves tiny positive losses and avoids overflow.

```python
from ml_scratch.pytorch.logistic_regression import BinaryLogisticRegression, LogisticTrainingConfig

model = BinaryLogisticRegression(1)
data = (((-1,), 0), ((1,), 1))
model.fit(data, LogisticTrainingConfig(max_epochs=100))
manual = model.gradient(data, positive_class_weight=3.7)
automatic = model.autograd_gradient(data, positive_class_weight=3.7)
assert model.predict([1]) == 1
```

Every fit starts from zero; shuffled mini-batches follow the scalar Python stream. Threshold
selection uses validation F1, then closest to 0.5, then lowest threshold. Equality predicts class 1.
Precision, recall, and F1 return zero for zero denominators. `predict_probabilities` accepts a
feature matrix; `predict_probability` takes one sample.

Autograd differentiates the weighted objective on parameter clones and leaves `.grad` fields
untouched. Tests compare weighted manual gradients with autograd, central differences, and scalar
Python, including a non-integer weight to detect accidental float32 rounding. The experiment uses
fixed training/validation/test splits, tunes the threshold only on validation, and reports test F1.
