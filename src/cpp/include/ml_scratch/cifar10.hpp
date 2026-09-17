#pragma once

#include "ml_scratch/dataset.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace ml_scratch {

struct Cifar10Split {
    LabeledDataset samples;
    std::size_t channels{3};
    std::size_t rows{32};
    std::size_t columns{32};
};

// Reads one CIFAR-10 binary batch file. Each of its 10000 records is a single label byte followed
// by 3072 pixel bytes: 1024 red, then 1024 green, then 1024 blue, each row-major. That is already
// the channel-major layout the convolutional network expects, so the bytes are rescaled into
// [0, 1] and copied straight through.
//
// `max_samples` of zero reads the whole file. Throws std::runtime_error when the file is missing,
// truncated, or carries a label outside 0..9.
[[nodiscard]] Cifar10Split load_cifar10_batch(const std::string& path, std::size_t max_samples = 0);

// Concatenates several batch files, stopping once `max_samples` records have been read.
[[nodiscard]] Cifar10Split load_cifar10(const std::vector<std::string>& paths,
                                        std::size_t max_samples = 0);

[[nodiscard]] const std::vector<std::string>& cifar10_class_names();

} // namespace ml_scratch
