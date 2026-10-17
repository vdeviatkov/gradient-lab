#include "ml_scratch/rnn.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <utility>

namespace ml_scratch {

void CharRnn::save(const std::string& path) const {
    std::ofstream stream{path};
    if (!stream) {
        throw std::runtime_error("cannot open checkpoint for writing: " + path);
    }
    stream << "ml_scratch_char_rnn 2\n"
           << "cell " << static_cast<int>(cell_) << '\n'
           << "vocabulary " << vocabulary_size_ << '\n'
           << "hidden " << hidden_size_ << '\n'
           << "parameters " << parameters_.size() << '\n'
           << std::setprecision(17);
    for (const double value : parameters_) {
        stream << value << '\n';
    }
    if (!stream) {
        throw std::runtime_error("failed while writing checkpoint: " + path);
    }
}

CharRnn CharRnn::load(const std::string& path) {
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

    expect("ml_scratch_char_rnn");
    const std::size_t version = read_size();
    if (version < 1 || version > 2) {
        throw std::runtime_error("unsupported checkpoint version in " + path);
    }
    // Version 1 predates the gated cells and is always an elman network.
    RecurrentCell cell = RecurrentCell::elman;
    if (version >= 2) {
        expect("cell");
        const std::size_t encoded = read_size();
        if (encoded > static_cast<std::size_t>(RecurrentCell::gru)) {
            throw std::runtime_error("unknown cell in checkpoint " + path);
        }
        cell = static_cast<RecurrentCell>(encoded);
    }
    expect("vocabulary");
    const std::size_t vocabulary_size = read_size();
    expect("hidden");
    const std::size_t hidden_size = read_size();
    if (vocabulary_size < 2 || hidden_size == 0) {
        throw std::runtime_error("checkpoint declares an invalid architecture: " + path);
    }
    CharRnn network{vocabulary_size, hidden_size, 0, cell};
    expect("parameters");
    const std::size_t count = read_size();
    if (count != network.parameter_count()) {
        throw std::runtime_error("checkpoint parameter count does not match its architecture: " +
                                 path);
    }
    std::vector<double> values(count, 0.0);
    for (double& value : values) {
        if (!(stream >> value)) {
            throw std::runtime_error("truncated checkpoint: " + path);
        }
    }
    network.set_parameters(values);
    return network;
}

} // namespace ml_scratch
