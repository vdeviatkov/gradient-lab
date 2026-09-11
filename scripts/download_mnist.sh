#!/usr/bin/env bash
# Downloads the MNIST dataset into data/mnist/ and decompresses it.
#
# The data is not committed: data/ is ignored by Git. The C++ loader reads the uncompressed IDX
# files, so this script gunzips what it fetches rather than teaching the loader about gzip.
set -euo pipefail

target_directory="${1:-data/mnist}"
mirror="${MNIST_MIRROR:-https://storage.googleapis.com/cvdf-datasets/mnist}"

files=(
    train-images-idx3-ubyte
    train-labels-idx1-ubyte
    t10k-images-idx3-ubyte
    t10k-labels-idx1-ubyte
)

mkdir -p "${target_directory}"
for file in "${files[@]}"; do
    if [[ -f "${target_directory}/${file}" ]]; then
        echo "already present: ${target_directory}/${file}"
        continue
    fi
    echo "downloading ${file}.gz"
    curl --fail --location --silent --show-error \
        --output "${target_directory}/${file}.gz" "${mirror}/${file}.gz"
    gunzip --force "${target_directory}/${file}.gz"
done

echo "MNIST is ready in ${target_directory}"
