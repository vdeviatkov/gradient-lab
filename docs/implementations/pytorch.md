# PyTorch conventions

[Implementation guides](README.md) · [Topics and status](../../README.md#topics) ·
[NumPy reference](numpy.md)

```bash
python -m pip install -e ".[dev,pytorch]"
python -m ml_scratch.pytorch.perceptron.experiment
python -m ml_scratch.pytorch.mlp.experiment
python -m ml_scratch.pytorch.linear_regression.experiment
python -m ml_scratch.pytorch.logistic_regression.experiment
python -m ml_scratch.pytorch.pca_kmeans.experiment
python -m ml_scratch.pytorch.decision_tree.experiment
python -m ml_scratch.pytorch.backpropagation.experiment
python -m ml_scratch.pytorch.softmax_mnist.experiment
python -m pytest topics/*/pytorch/tests
```

Topics 01–08 own implementations under `pytorch/`, installed as `ml_scratch.pytorch.<model>`
through the explicit [package map](../../pyproject.toml). Their `ml-pytorch-*` entry points invoke
the same experiments. Pure-Python and NumPy imports do not require Torch. The existing `pytorch`
extra also installs torchvision for later image topics; these eight implementations use Torch alone.

These are CPU float64 educational models. Inputs are converted to CPU float64 and detached at
the boundary; externally supplied autograd graphs do not flow through the public data API. Model
calculations use Torch tensors throughout, without converting computation to NumPy or delegating
to a scalar model. Seeds and scalar configuration/result value types are shared with Python;
reusing fixtures and report records does not reuse the learning algorithm.

Supervised models accept `(features, target)` pair datasets. PCA and k-means accept rectangular
matrices. Single-sample prediction methods return Python numbers; batch predictions, projections,
and parameter-gradient arrays return tensors. MLP parameter snapshots use immutable Python tuples.
The perceptron, MLP, and linear regression continue current parameters on fit; logistic regression
resets to zero on every fit, matching the existing reference contracts.

Manual gradients train the MLP, linear regression, logistic regression, the general backpropagation
network, and softmax regression under `torch.no_grad()`. Their `autograd_gradient` methods
independently build differentiable forward objectives on cloned parameters using
`torch.enable_grad()` and `torch.autograd.grad`. They do not call manual gradient code, mutate model
parameters, or accumulate `.grad` fields. Tests also use central differences and scalar derivatives.
Perceptron updates, PCA, k-means, and decision trees remain explicit discrete algorithms. No
`nn.Linear`, optimizer, linear-system solver, eigensolver, clustering or tree estimator replaces an
algorithm being taught; library solvers are only test oracles.

Local `random.Random` streams preserve initialization and shuffling across Python, NumPy, and
Torch. No global seed, default dtype, or thread setting is modified. Floating-point reduction
order can change low bits, ties, or stopping decisions; compare aligned parameters with tolerances,
not universal bitwise identities. Only CPU behavior is tested; no GPU, MPS, or speedup claim is made.
The MLP width sweep defaults to ten seeds; use `--seeds 50` for the scalar experiment's budget.

Run the whole repository's Python suite after installing `.[dev,numpy,pytorch]`. CI does the same.
Topic reports record commands, seeds, environment, and actual measurements separately from runtime
benchmarks. See the [shared tensor primitives](../../shared/python/ml_scratch/pytorch/validation.py)
for input validation and stable sigmoid/BCE calculations.
