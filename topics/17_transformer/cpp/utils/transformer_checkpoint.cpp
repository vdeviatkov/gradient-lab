#include "ml_scratch/transformer.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <utility>

namespace ml_scratch {

void CharTransformer::save(const std::string& path) const {
    std::ofstream stream{path};
    if (!stream) {
        throw std::runtime_error("cannot open checkpoint for writing: " + path);
    }
    stream << "ml_scratch_char_transformer 1\n"
           << "vocabulary " << config_.vocabulary_size << '\n'
           << "context " << config_.context_length << '\n'
           << "model " << config_.model_size << '\n'
           << "heads " << config_.heads << '\n'
           << "layers " << config_.layers << '\n'
           << "feed_forward " << config_.feed_forward_size << '\n'
           << "positional " << (config_.positional_encoding ? 1 : 0) << '\n'
           << "relative_bias " << (config_.relative_position_bias ? 1 : 0) << '\n'
           << "parameters " << parameters_.size() << '\n'
           << std::setprecision(17);
    for (const double value : parameters_) {
        stream << value << '\n';
    }
    if (!stream) {
        throw std::runtime_error("failed while writing checkpoint: " + path);
    }
}

CharTransformer CharTransformer::load(const std::string& path) {
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
    expect("ml_scratch_char_transformer");
    if (read_size() != 1) {
        throw std::runtime_error("unsupported checkpoint version in " + path);
    }
    TransformerConfig config{};
    expect("vocabulary");
    config.vocabulary_size = read_size();
    expect("context");
    config.context_length = read_size();
    expect("model");
    config.model_size = read_size();
    expect("heads");
    config.heads = read_size();
    expect("layers");
    config.layers = read_size();
    expect("feed_forward");
    config.feed_forward_size = read_size();
    expect("positional");
    config.positional_encoding = read_size() != 0;
    expect("relative_bias");
    config.relative_position_bias = read_size() != 0;
    // The constructor re-validates the architecture.
    CharTransformer network{config};
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
