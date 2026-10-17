# Scripts

[Setup](../docs/getting-started.md) · [Testing](../docs/testing.md) ·
[Experiments](../experiments/README.md)

Run these commands from the repository root. Download scripts use Bash and `curl`; decompression
also requires `gunzip` for MNIST and `tar` for CIFAR-10.

| Script | Default destination / behavior | Overrides |
|---|---|---|
| [download_mnist.sh](download_mnist.sh) | Uncompressed IDX files in `data/mnist/` | First argument: directory; `MNIST_MIRROR`: mirror base URL |
| [download_cifar10.sh](download_cifar10.sh) | Binary batches in `data/cifar10/` | First argument: directory; `CIFAR10_URL`: archive URL |
| [download_tiny_shakespeare.sh](download_tiny_shakespeare.sh) | `data/tiny_shakespeare/input.txt` | First argument: directory; `TINY_SHAKESPEARE_URL`: corpus URL |
| [benchmark_build.sh](benchmark_build.sh) | Measures a clean Release build into `results/benchmarks/cpp_build.json` | First argument: temporary build directory; `ML_SCRATCH_BENCHMARK_DIR`, `ML_SCRATCH_BUILD_JOBS` |
| [check_docs.py](check_docs.py) | Checks repository-local Markdown destinations and heading anchors | `--root`: repository checkout to inspect |

`benchmark_build.sh` removes and recreates its build directory and removes it again after timing;
use a dedicated directory, not an existing development build. It also requires `bc` and a `date`
implementation supporting nanosecond timestamps. See
[experiment 19](../experiments/19_benchmarks/README.md) for measurement methodology.

The documentation checker uses Python 3.11+ and Git, with no runtime dependency or network access:

```bash
python scripts/check_docs.py
```

Dataset files, build products, and generated measurements are ignored by Git. Downloading the data
is separate from running tests, which use small generated fixtures.
