# Triton conventions

[Implementation guides](README.md) · [Topics and status](../../README.md#topics) ·
[PyTorch reference](pytorch.md)

```bash
# Linux with a CUDA GPU; Triton publishes Linux wheels only.
python -m pip install -e ".[dev,numpy,triton]"
python -m ml_scratch.triton.softmax_mnist.experiment
python -m pytest topics/*/triton/tests

# Linux without a GPU: run the same kernels on Triton's CPU interpreter.
TRITON_INTERPRET=1 python -m pytest topics/*/triton/tests
```

Topic 08 owns the first Triton implementation, under `triton/`, installed as
`ml_scratch.triton.<model>` through the explicit [package map](../../pyproject.toml). Triton is
used where a hand-written kernel teaches something the array backends hide: tiling, explicit
loads and masks, data movement, and fusion. The other backends reach those topics only through
library calls. Torch supplies device tensors, memory allocation, and host-side bookkeeping.
Every model computation runs inside the topic's own `@triton.jit` kernels; no Torch operator
replaces a kernel's algorithm.

Kernels compute in float64, so their results can be compared with the scalar, NumPy, and PyTorch
backends to rounding error rather than to float32 tolerances. Consumer GPUs run float64 at a
fraction of their float32 rate, which is acceptable for verification but makes these timings
unrepresentative of production kernels. No speedup is claimed. Kernels avoid atomics where an
owner-per-output reduction is possible, so results are bit-reproducible run to run.

Two Triton-specific rules apply:
- **Never pass a precision-sensitive Python `float` as a kernel argument.** Triton passes it as
  fp32. Compute reciprocals in-kernel from integer sizes, and pass float64 hyperparameters in a
  small device array.
- **Keep loop trip counts compile-time constants.** Pass a `constexpr` bound and mask against the
  runtime size. This lets similar launches share a compilation, and it keeps the kernels runnable
  under `TRITON_INTERPRET=1`, whose interpreter cannot use a runtime integer as a loop bound with
  current NumPy.

Tests skip unless Triton is importable and either a CUDA device or `TRITON_INTERPRET=1` is
present. GitHub's runners have no GPU, so CI runs the Triton tests under the interpreter.
Measured results are recorded from real GPU runs, and topic READMEs name the GPU, driver, CUDA,
Torch, and Triton versions.
