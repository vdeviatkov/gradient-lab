#!/usr/bin/env bash
# Downloads the Tiny Shakespeare corpus into data/tiny_shakespeare/input.txt.
#
# The data is not committed: data/ is ignored by Git. The corpus is one 1.1 MB plain-text file of
# 65 distinct characters, the standard character-level modeling text; the sequence milestones
# (RNN, LSTM and GRU, Transformer) all read the same fixed prefix of it so that they can be compared.
set -euo pipefail

target_directory="${1:-data/tiny_shakespeare}"
url="${TINY_SHAKESPEARE_URL:-https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt}"

mkdir -p "${target_directory}"
if [[ -f "${target_directory}/input.txt" ]]; then
    echo "already present: ${target_directory}/input.txt"
else
    echo "downloading input.txt"
    curl --fail --location --silent --show-error --output "${target_directory}/input.txt" "${url}"
fi

echo "Tiny Shakespeare is ready in ${target_directory}"
