#include "ml_scratch/mnist.hpp"

#include <fstream>
#include <stdexcept>

namespace ml_scratch {
namespace {

constexpr std::uint8_t unsigned_byte_type = 0x08;

std::uint32_t read_big_endian(std::istream& stream, const std::string& path) {
    unsigned char bytes[4];
    stream.read(reinterpret_cast<char*>(bytes), 4);
    if (!stream) {
        throw std::runtime_error("truncated IDX header in " + path);
    }
    // IDX is big-endian regardless of the host, so the bytes are combined explicitly rather than
    // copied over an integer.
    return static_cast<std::uint32_t>(bytes[0]) << 24 | static_cast<std::uint32_t>(bytes[1]) << 16 |
           static_cast<std::uint32_t>(bytes[2]) << 8 | static_cast<std::uint32_t>(bytes[3]);
}

} // namespace

IdxData read_idx_file(const std::string& path) {
    std::ifstream stream{path, std::ios::binary};
    if (!stream) {
        throw std::runtime_error("cannot open IDX file " + path);
    }

    const std::uint32_t magic = read_big_endian(stream, path);
    // The magic number is two zero bytes, an element-type byte, and a dimension count.
    if ((magic >> 16) != 0) {
        throw std::runtime_error("not an IDX file: " + path);
    }
    const auto element_type = static_cast<std::uint8_t>((magic >> 8) & 0xFF);
    const auto dimension_count = static_cast<std::uint8_t>(magic & 0xFF);
    if (element_type != unsigned_byte_type) {
        throw std::runtime_error("unsupported IDX element type in " + path +
                                 "; only unsigned bytes are supported");
    }
    if (dimension_count == 0) {
        throw std::runtime_error("IDX file declares no dimensions: " + path);
    }

    IdxData data;
    data.dimensions.reserve(dimension_count);
    std::size_t total = 1;
    for (std::uint8_t dimension = 0; dimension < dimension_count; ++dimension) {
        const auto size = static_cast<std::size_t>(read_big_endian(stream, path));
        if (size == 0) {
            throw std::runtime_error("IDX file declares an empty dimension: " + path);
        }
        data.dimensions.push_back(size);
        total *= size;
    }

    data.values.resize(total);
    stream.read(reinterpret_cast<char*>(data.values.data()), static_cast<std::streamsize>(total));
    if (stream.gcount() != static_cast<std::streamsize>(total)) {
        throw std::runtime_error("truncated IDX payload in " + path);
    }
    return data;
}

MnistSplit load_mnist(const std::string& images_path, const std::string& labels_path,
                      const std::size_t max_samples) {
    const IdxData images = read_idx_file(images_path);
    const IdxData labels = read_idx_file(labels_path);

    if (images.dimensions.size() != 3) {
        throw std::runtime_error("expected a three-dimensional image file: " + images_path);
    }
    if (labels.dimensions.size() != 1) {
        throw std::runtime_error("expected a one-dimensional label file: " + labels_path);
    }
    if (images.dimensions[0] != labels.dimensions[0]) {
        throw std::runtime_error("image and label counts disagree");
    }

    MnistSplit split{{}, images.dimensions[1], images.dimensions[2]};
    const std::size_t pixel_count = split.rows * split.columns;
    const std::size_t count =
        max_samples == 0 ? images.dimensions[0] : std::min(max_samples, images.dimensions[0]);

    split.samples.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        if (labels.values[index] > 9) {
            throw std::runtime_error("MNIST label outside 0..9");
        }
        std::vector<double> features(pixel_count);
        const std::size_t offset = index * pixel_count;
        for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
            features[pixel] = static_cast<double>(images.values[offset + pixel]) / 255.0;
        }
        split.samples.push_back({std::move(features), labels.values[index]});
    }
    return split;
}

} // namespace ml_scratch
