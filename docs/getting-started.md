# Getting started

[Documentation](README.md) · [All experiments](../README.md#topics)

Run commands from the repository root unless a topic README says otherwise.

## Python

Requires Python 3.11 or newer. The pure-Python models use only the standard library.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m ml_scratch.pure.perceptron.experiment
python -m ml_scratch.pure.mlp.experiment
python -m ml_scratch.pure.linear_regression.experiment
python -m ml_scratch.pure.logistic_regression.experiment
```

Each topic's Python README links its source and documents the module and installed CLI commands.
See the [pure-Python conventions](implementations/pure-python.md) for the package layout.
Use each command's `--help` to inspect its options.

Optional dependencies are available with `python -m pip install -e ".[numpy]"`,
`python -m pip install -e ".[pytorch]"`, and `python -m pip install -e ".[plotting]"`.
NumPy implementations for topics 01–07 are available; for example,
`python -m ml_scratch.numpy.perceptron.experiment`. See [NumPy conventions](implementations/numpy.md)
for all commands. PyTorch versions for the same topics are available under `ml_scratch.pytorch`; see
[PyTorch conventions](implementations/pytorch.md). To run the full Python test suite, install
`.[dev,numpy,pytorch]`.
Check [implementation status](../README.md#topics) for other backends.

## C++

Requires CMake 3.20 or newer and a compiler supporting C++20. No third-party library is required.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/cpp_perceptron_logic_gates
./build/cpp_logistic_regression
```

For a multi-configuration generator, add `--config Release` to the build command and
`--build-config Release` to CTest; executable locations follow the generator’s configuration
layout. See the [C++ guide](implementations/cpp.md) for library boundaries and source locations.

## Datasets

Small introductory experiments and every automated test run without downloading a dataset.
The MNIST, CIFAR-10, and character-model experiments require local data:

| Dataset | Download command | Used by |
|---|---|---|
| MNIST | `./scripts/download_mnist.sh` | Topics 08, 09, 12, 13, 14 |
| CIFAR-10 | `./scripts/download_cifar10.sh` | Topic 13 |
| Tiny Shakespeare | `./scripts/download_tiny_shakespeare.sh` | Topics 15–17 |

The scripts write under ignored `data/` directories. Each topic documents its path overrides and
configuration; see the [script reference](../scripts/README.md). Checkpoints, generated figures,
and benchmark output stay local. Follow the [results policy](../results/README.md) when publishing
measurements.

## Verify the checkout

Follow [testing and quality checks](testing.md). The Python and C++ suites use deterministic,
small fixtures and do not require GPUs or downloaded data.
