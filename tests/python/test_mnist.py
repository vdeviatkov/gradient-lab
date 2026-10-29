import pytest

from ml_scratch.pure.mnist import (
    FILES,
    load_mnist,
    load_mnist_directory,
    mnist_directory,
    read_idx,
    write_idx,
)


def write_split(directory, prefix, images, labels, rows=2, columns=3):
    write_idx(
        directory / f"{prefix}-images-idx3-ubyte", (len(images), rows, columns), b"".join(images)
    )
    write_idx(directory / f"{prefix}-labels-idx1-ubyte", (len(labels),), labels)


def test_idx_round_trip_and_mnist_pairing(tmp_path):
    images = [bytes(range(6)), bytes([255] * 6), bytes(6)]
    write_split(tmp_path, "train", images, [3, 1, 9])
    write_split(tmp_path, "t10k", images[:1], [0])
    idx = read_idx(tmp_path / FILES[0])
    assert idx.dimensions == (3, 2, 3) and idx.values == b"".join(images)
    split = load_mnist(tmp_path / FILES[0], tmp_path / FILES[1])
    assert split.images == tuple(images) and split.labels == (3, 1, 9)
    assert (split.rows, split.columns) == (2, 3)
    assert load_mnist(tmp_path / FILES[0], tmp_path / FILES[1], max_samples=2).labels == (3, 1)
    train, test = load_mnist_directory(tmp_path)
    assert train == split and test.labels == (0,)


def test_directory_override(monkeypatch, tmp_path):
    monkeypatch.setenv("ML_SCRATCH_MNIST_DIR", str(tmp_path))
    assert mnist_directory() == tmp_path
    with pytest.raises(FileNotFoundError, match="download_mnist.sh"):
        load_mnist_directory()


@pytest.mark.parametrize(
    "raw",
    [
        b"\x00\x00",  # truncated header
        b"\x01\x00\x08\x01\x00\x00\x00\x01\x00",  # not an IDX magic number
        b"\x00\x00\x0d\x01\x00\x00\x00\x01\x00",  # float elements
        b"\x00\x00\x08\x00",  # no dimensions
        b"\x00\x00\x08\x01\x00\x00\x00\x00",  # empty dimension
        b"\x00\x00\x08\x01\x00\x00\x00\x02\x00",  # one value short
        b"\x00\x00\x08\x01\x00\x00\x00",  # truncated sizes
    ],
)
def test_malformed_idx_files_are_rejected(tmp_path, raw):
    path = tmp_path / "bad"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        read_idx(path)


def test_mismatched_or_invalid_pairs_are_rejected(tmp_path):
    write_split(tmp_path, "train", [bytes(6)] * 2, [1])
    with pytest.raises(ValueError, match="different sample counts"):
        load_mnist(tmp_path / FILES[0], tmp_path / FILES[1])
    write_split(tmp_path, "train", [bytes(6)], [10])
    with pytest.raises(ValueError, match="digits"):
        load_mnist(tmp_path / FILES[0], tmp_path / FILES[1])
    with pytest.raises(ValueError, match="three-dimensional"):
        load_mnist(tmp_path / FILES[1], tmp_path / FILES[1])
    write_split(tmp_path, "train", [bytes(6)], [1])
    with pytest.raises(ValueError):
        load_mnist(tmp_path / FILES[0], tmp_path / FILES[1], max_samples=-1)
