"""MNIST's IDX files read with the standard library, mirroring the C++ loader's checks.

Images stay raw bytes here. Each backend scales a pixel with ``byte / 255.0``, the C++ loader's
single preprocessing step, so every backend sees bit-identical pixel values.
"""

import os
from dataclasses import dataclass
from pathlib import Path

UNSIGNED_BYTE = 0x08
FILES = (
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
)
MISSING_DATA_HELP = (
    "Download it first (the data is not committed):\n"
    "    ./scripts/download_mnist.sh\n"
    "or point ML_SCRATCH_MNIST_DIR at a directory holding the uncompressed IDX files."
)


@dataclass(frozen=True)
class IdxData:
    # {count, rows, columns} for images, {count} for labels.
    dimensions: tuple[int, ...]
    values: bytes


@dataclass(frozen=True)
class MnistSplit:
    # One bytes object of rows * columns pixels per image, values 0..255.
    images: tuple[bytes, ...]
    labels: tuple[int, ...]
    rows: int
    columns: int


def mnist_directory():
    return Path(os.environ.get("ML_SCRATCH_MNIST_DIR", "data/mnist"))


def read_idx(path):
    """Decode an unsigned-byte IDX file: a big-endian magic number, sizes, then raw values."""
    data = Path(path).read_bytes()
    if len(data) < 4:
        raise ValueError(f"truncated IDX header in {path}")
    # The magic number is two zero bytes, an element-type byte, and a dimension count.
    if data[0] != 0 or data[1] != 0:
        raise ValueError(f"not an IDX file: {path}")
    if data[2] != UNSIGNED_BYTE:
        raise ValueError(f"unsupported IDX element type in {path}; only unsigned bytes")
    count = data[3]
    if count == 0:
        raise ValueError(f"IDX file declares no dimensions: {path}")
    header = 4 + 4 * count
    if len(data) < header:
        raise ValueError(f"truncated IDX header in {path}")
    dimensions = tuple(int.from_bytes(data[4 + 4 * i : 8 + 4 * i], "big") for i in range(count))
    if 0 in dimensions:
        raise ValueError(f"IDX file declares an empty dimension: {path}")
    total = 1
    for size in dimensions:
        total *= size
    if len(data) - header != total:
        raise ValueError(f"IDX file {path} holds {len(data) - header} values, expected {total}")
    return IdxData(dimensions, data[header:])


def load_mnist(images_path, labels_path, max_samples=0):
    """Pair an image file with a label file; ``max_samples`` of zero loads everything."""
    images, labels = read_idx(images_path), read_idx(labels_path)
    if len(images.dimensions) != 3 or len(labels.dimensions) != 1:
        raise ValueError("expected a three-dimensional image file and a one-dimensional label file")
    count, rows, columns = images.dimensions
    if labels.dimensions[0] != count:
        raise ValueError("image and label files hold different sample counts")
    if any(label > 9 for label in labels.values):
        raise ValueError("MNIST labels must be digits 0..9")
    if type(max_samples) is not int or max_samples < 0:
        raise ValueError("max_samples must be a non-negative integer")
    count = min(count, max_samples) if max_samples else count
    size = rows * columns
    pixels = images.values
    return MnistSplit(
        tuple(pixels[i * size : (i + 1) * size] for i in range(count)),
        tuple(labels.values[:count]),
        rows,
        columns,
    )


def load_mnist_directory(directory=None, max_samples=0):
    """Return the official (train, test) splits, or explain how to download them."""
    directory = Path(directory) if directory is not None else mnist_directory()
    try:
        return (
            load_mnist(directory / FILES[0], directory / FILES[1], max_samples),
            load_mnist(directory / FILES[2], directory / FILES[3], max_samples),
        )
    except (OSError, ValueError) as error:
        raise FileNotFoundError(
            f"could not load MNIST from {directory}: {error}\n\n{MISSING_DATA_HELP}"
        ) from error


def write_idx(path, dimensions, values):
    """Write an unsigned-byte IDX file; used to build small fixtures for tests."""
    header = bytes((0, 0, UNSIGNED_BYTE, len(dimensions)))
    header += b"".join(size.to_bytes(4, "big") for size in dimensions)
    Path(path).write_bytes(header + bytes(values))
