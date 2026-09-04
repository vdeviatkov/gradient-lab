# Experiment 02: two-layer MLP for XOR

The C++ implementation uses a `2 → 4 → 1` network with tanh hidden activations and a sigmoid
output. It trains with full-batch gradient descent on binary cross-entropy using gradients written
directly from the chain rule. The fixed truth table, initialization seed, epoch budget, and loss
threshold make the experiment reproducible within a given standard-library/toolchain environment.

Build from the repository root and run:

```bash
cmake -S . -B build
cmake --build build
./build/cpp_xor_mlp
```

The executable exits unsuccessfully unless all four XOR predictions are correct and the measured
loss reaches the configured threshold. A reusable backpropagation abstraction and numerical
gradient checking remain a later roadmap milestone; NumPy, PyTorch, and pure-Python XOR MLP
versions are also still planned.
