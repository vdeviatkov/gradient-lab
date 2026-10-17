#include "ml_scratch/neural_network.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <utility>

namespace ml_scratch {

void FeedForwardNetwork::save(const std::string& path) const {
    std::ofstream stream{path};
    if (!stream) {
        throw std::runtime_error("cannot open checkpoint for writing: " + path);
    }
    save(stream, path);
}

void FeedForwardNetwork::save(std::ostream& stream, const std::string& path) const {
    stream << "ml_scratch_feedforward 3\n"
           << "loss " << static_cast<int>(loss_) << '\n'
           << "layers " << layers_.size() << '\n'
           << std::setprecision(17);
    for (const DenseLayer& layer : layers_) {
        stream << layer.input_size << ' ' << layer.output_size << ' '
               << static_cast<int>(layer.activation) << ' ' << static_cast<int>(layer.normalization)
               << ' ' << layer.dropout_rate << ' ' << (layer.residual ? 1 : 0) << '\n';
    }

    const std::vector<double> flat = parameters();
    stream << "parameters " << flat.size() << '\n';
    // 17 significant digits round-trip an IEEE-754 double exactly.
    for (const double value : flat) {
        stream << value << '\n';
    }

    // Batch normalization's running statistics are not parameters, but inference is wrong without
    // them, so a checkpoint that omitted them would restore a different function.
    const std::vector<double> statistics = running_statistics();
    stream << "statistics " << statistics.size() << '\n';
    for (const double value : statistics) {
        stream << value << '\n';
    }
    if (!stream) {
        throw std::runtime_error("failed while writing checkpoint: " + path);
    }
}

FeedForwardNetwork FeedForwardNetwork::load(const std::string& path) {
    std::ifstream stream{path};
    if (!stream) {
        throw std::runtime_error("cannot open checkpoint: " + path);
    }
    return load(stream, path);
}

FeedForwardNetwork FeedForwardNetwork::load(std::istream& stream, const std::string& path) {
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

    expect("ml_scratch_feedforward");
    const std::size_t version = read_size();
    if (version < 1 || version > 3) {
        throw std::runtime_error("unsupported checkpoint version in " + path);
    }
    expect("loss");
    const std::size_t loss_value = read_size();
    if (loss_value > static_cast<std::size_t>(Loss::softmax_cross_entropy)) {
        throw std::runtime_error("unknown loss in checkpoint " + path);
    }
    expect("layers");
    const std::size_t layer_count = read_size();
    if (layer_count == 0) {
        throw std::runtime_error("checkpoint declares no layers: " + path);
    }

    std::vector<DenseLayer> layers;
    layers.reserve(layer_count);
    for (std::size_t index = 0; index < layer_count; ++index) {
        const std::size_t input_size = read_size();
        const std::size_t output_size = read_size();
        const std::size_t activation = read_size();
        if (activation > static_cast<std::size_t>(Activation::leaky_rectified_linear)) {
            throw std::runtime_error("unknown activation in checkpoint " + path);
        }

        // Version 1 predates normalization and dropout, and version 2 predates the skip, so the
        // absent fields take their defaults.
        Normalization normalization = Normalization::none;
        double dropout_rate = 0.0;
        bool residual = false;
        if (version >= 2) {
            const std::size_t encoded = read_size();
            if (encoded > static_cast<std::size_t>(Normalization::layer)) {
                throw std::runtime_error("unknown normalization in checkpoint " + path);
            }
            normalization = static_cast<Normalization>(encoded);
            if (!(stream >> dropout_rate) || !std::isfinite(dropout_rate) || dropout_rate < 0.0 ||
                dropout_rate >= 1.0) {
                throw std::runtime_error("invalid dropout rate in checkpoint " + path);
            }
        }
        if (version >= 3) {
            residual = read_size() != 0;
        }
        layers.push_back({input_size, output_size, static_cast<Activation>(activation),
                          normalization, dropout_rate, residual});
    }

    // The constructor re-validates the architecture, so a corrupted shape is rejected here.
    FeedForwardNetwork network{std::move(layers), static_cast<Loss>(loss_value)};

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

    if (version >= 2) {
        expect("statistics");
        const std::size_t statistics_count = read_size();
        if (statistics_count != network.running_statistics().size()) {
            throw std::runtime_error("checkpoint statistics count does not match its "
                                     "architecture: " +
                                     path);
        }
        std::vector<double> statistics(statistics_count, 0.0);
        for (double& value : statistics) {
            if (!(stream >> value)) {
                throw std::runtime_error("truncated checkpoint: " + path);
            }
        }
        network.set_running_statistics(statistics);
    }
    return network;
}

} // namespace ml_scratch
