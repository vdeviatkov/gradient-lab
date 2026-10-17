#include "ml_scratch/gan.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <utility>

namespace ml_scratch {

void ConditionalGan::save(const std::string& path) const {
    std::ofstream stream{path};
    if (!stream) {
        throw std::runtime_error("cannot open checkpoint for writing: " + path);
    }
    stream << "ml_scratch_conditional_gan 1\n"
           << "noise " << noise_size_ << '\n'
           << "classes " << class_count_ << '\n';
    generator_.save(stream, path);
    discriminator_.save(stream, path);
    if (!stream) {
        throw std::runtime_error("failed while writing checkpoint: " + path);
    }
}

ConditionalGan ConditionalGan::load(const std::string& path) {
    std::ifstream stream{path};
    if (!stream) {
        throw std::runtime_error("cannot open checkpoint: " + path);
    }
    const auto expect = [&stream, &path](const std::string& keyword) {
        std::string token;
        if (!(stream >> token) || token != keyword) {
            throw std::runtime_error("malformed checkpoint " + path + ": expected " + keyword);
        }
    };
    const auto read_size = [&stream, &path] {
        long long value = 0;
        if (!(stream >> value) || value < 0) {
            throw std::runtime_error("malformed checkpoint " + path + ": expected a size");
        }
        return static_cast<std::size_t>(value);
    };

    expect("ml_scratch_conditional_gan");
    if (read_size() != 1) {
        throw std::runtime_error("unsupported checkpoint version in " + path);
    }
    expect("noise");
    const std::size_t noise_size = read_size();
    expect("classes");
    const std::size_t class_count = read_size();
    if (noise_size == 0 || class_count == 0) {
        throw std::runtime_error("checkpoint declares an empty noise or class size: " + path);
    }
    FeedForwardNetwork generator = FeedForwardNetwork::load(stream, path);
    FeedForwardNetwork discriminator = FeedForwardNetwork::load(stream, path);
    // The private constructor re-checks that the two networks fit together.
    return ConditionalGan{noise_size, class_count, std::move(generator), std::move(discriminator)};
}

} // namespace ml_scratch
