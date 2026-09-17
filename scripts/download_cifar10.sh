#!/usr/bin/env bash
# Downloads the CIFAR-10 binary distribution into data/cifar10/ and unpacks it.
#
# The data is not committed: data/ is ignored by Git. The C++ loader reads the raw binary batch
# files, so this script unpacks the archive rather than teaching the loader about tar.
set -euo pipefail

target_directory="${1:-data/cifar10}"
url="${CIFAR10_URL:-https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz}"

if [[ -f "${target_directory}/test_batch.bin" ]]; then
    echo "already present: ${target_directory}"
    exit 0
fi

mkdir -p "${target_directory}"
archive="${target_directory}/cifar-10-binary.tar.gz"
echo "downloading cifar-10-binary.tar.gz (about 170 MB)"
# --continue-at resumes a partial file, so an interrupted transfer can be retried by rerunning.
# --retry covers transient failures on a download this size.
curl --fail --location --silent --show-error --continue-at - \
    --retry 5 --retry-delay 3 --retry-connrefused \
    --output "${archive}" "${url}"
# The archive holds a cifar-10-batches-bin/ directory; strip it so the batch files land directly.
tar --extract --gzip --file "${archive}" --directory "${target_directory}" --strip-components=1
rm -f "${archive}"

echo "CIFAR-10 is ready in ${target_directory}"
ls -1 "${target_directory}"
