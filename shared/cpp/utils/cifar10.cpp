#include "ml_scratch/cifar10.hpp"

#include <fstream>
#include <stdexcept>

namespace ml_scratch {
namespace {

constexpr std::size_t channels = 3;
constexpr std::size_t rows = 32;
constexpr std::size_t columns = 32;
constexpr std::size_t pixels = channels * rows * columns;
constexpr std::size_t record_size = 1 + pixels;

} // namespace

Cifar10Split load_cifar10_batch(const std::string& path, const std::size_t max_samples) {
    std::ifstream stream{path, std::ios::binary};
    if (!stream) {
        throw std::runtime_error("cannot open CIFAR-10 batch " + path);
    }

    Cifar10Split split;
    std::vector<unsigned char> record(record_size);
    while (max_samples == 0 || split.samples.size() < max_samples) {
        stream.read(reinterpret_cast<char*>(record.data()),
                    static_cast<std::streamsize>(record_size));
        const auto read = static_cast<std::size_t>(stream.gcount());
        if (read == 0) {
            break; // A clean end of file.
        }
        if (read != record_size) {
            throw std::runtime_error("truncated CIFAR-10 record in " + path);
        }
        if (record[0] > 9) {
            throw std::runtime_error("CIFAR-10 label outside 0..9 in " + path);
        }

        std::vector<double> features(pixels);
        for (std::size_t index = 0; index < pixels; ++index) {
            features[index] = static_cast<double>(record[index + 1]) / 255.0;
        }
        split.samples.push_back({std::move(features), record[0]});
    }

    if (split.samples.empty()) {
        throw std::runtime_error("CIFAR-10 batch is empty: " + path);
    }
    return split;
}

Cifar10Split load_cifar10(const std::vector<std::string>& paths, const std::size_t max_samples) {
    if (paths.empty()) {
        throw std::runtime_error("no CIFAR-10 batch files given");
    }

    Cifar10Split combined;
    for (const std::string& path : paths) {
        const std::size_t remaining = max_samples == 0 ? 0 : max_samples - combined.samples.size();
        if (max_samples != 0 && remaining == 0) {
            break;
        }
        Cifar10Split batch = load_cifar10_batch(path, remaining);
        combined.samples.insert(combined.samples.end(),
                                std::make_move_iterator(batch.samples.begin()),
                                std::make_move_iterator(batch.samples.end()));
    }
    return combined;
}

const std::vector<std::string>& cifar10_class_names() {
    static const std::vector<std::string> names{"airplane", "automobile", "bird",  "cat",  "deer",
                                                "dog",      "frog",       "horse", "ship", "truck"};
    return names;
}

} // namespace ml_scratch
