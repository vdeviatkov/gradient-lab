#pragma once

#include "ml_scratch/dataset.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ml_scratch {

// One decoded IDX file. MNIST stores its images and labels in this format: a big-endian magic
// number, one big-endian size per dimension, then the raw values.
struct IdxData {
    // For MNIST images this is {count, rows, columns}; for labels it is {count}.
    std::vector<std::size_t> dimensions;
    std::vector<std::uint8_t> values;

    [[nodiscard]] std::size_t value_count() const noexcept { return values.size(); }
};

// Reads an uncompressed IDX file of unsigned bytes, the only element type MNIST uses. Throws
// std::runtime_error when the file is missing, truncated, or not an unsigned-byte IDX file.
[[nodiscard]] IdxData read_idx_file(const std::string& path);

struct MnistSplit {
    LabeledDataset samples;
    std::size_t rows;
    std::size_t columns;
};

// Pairs an image file with a label file and scales each pixel from its stored 0..255 range into
// [0, 1]. That rescaling is the whole preprocessing step, and it is applied identically to every
// split. `max_samples` of zero loads the entire file.
[[nodiscard]] MnistSplit load_mnist(const std::string& images_path, const std::string& labels_path,
                                    std::size_t max_samples = 0);

} // namespace ml_scratch
