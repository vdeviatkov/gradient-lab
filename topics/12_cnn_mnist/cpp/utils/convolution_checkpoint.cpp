#include "ml_scratch/convolution.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <utility>

namespace ml_scratch {

void ConvolutionalNetwork::save(const std::string& path) const {
    std::ofstream stream{path};
    if (!stream) {
        throw std::runtime_error("cannot open checkpoint for writing: " + path);
    }

    const TensorShape input = input_shape();
    stream << "ml_scratch_convolutional 1\n"
           << "loss " << static_cast<int>(loss_) << '\n'
           << "input " << input.channels << ' ' << input.height << ' ' << input.width << '\n'
           << "layers " << layers_.size() << '\n';
    for (const ConvLayerSpec& layer : layers_) {
        stream << static_cast<int>(layer.kind) << ' ' << static_cast<int>(layer.activation) << ' '
               << layer.filters << ' ' << layer.kernel_size << ' ' << layer.stride << ' '
               << layer.padding << ' ' << layer.pool_size << ' ' << layer.pool_stride << ' '
               << layer.units << ' ' << (layer.residual ? 1 : 0) << '\n';
    }

    const std::vector<double> flat = parameters();
    stream << "parameters " << flat.size() << '\n' << std::setprecision(17);
    for (const double value : flat) {
        stream << value << '\n';
    }
    if (!stream) {
        throw std::runtime_error("failed while writing checkpoint: " + path);
    }
}

ConvolutionalNetwork ConvolutionalNetwork::load(const std::string& path) {
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

    expect("ml_scratch_convolutional");
    if (read_size() != 1) {
        throw std::runtime_error("unsupported checkpoint version in " + path);
    }
    expect("loss");
    const std::size_t loss_value = read_size();
    if (loss_value > static_cast<std::size_t>(Loss::softmax_cross_entropy)) {
        throw std::runtime_error("unknown loss in checkpoint " + path);
    }
    expect("input");
    TensorShape input;
    input.channels = read_size();
    input.height = read_size();
    input.width = read_size();
    expect("layers");
    const std::size_t layer_count = read_size();
    if (layer_count == 0) {
        throw std::runtime_error("checkpoint declares no layers: " + path);
    }

    std::vector<ConvLayerSpec> layers;
    layers.reserve(layer_count);
    for (std::size_t index = 0; index < layer_count; ++index) {
        ConvLayerSpec layer;
        const std::size_t kind = read_size();
        const std::size_t activation = read_size();
        if (kind > static_cast<std::size_t>(ConvLayerKind::dense)) {
            throw std::runtime_error("unknown layer kind in checkpoint " + path);
        }
        if (activation > static_cast<std::size_t>(Activation::leaky_rectified_linear)) {
            throw std::runtime_error("unknown activation in checkpoint " + path);
        }
        layer.kind = static_cast<ConvLayerKind>(kind);
        layer.activation = static_cast<Activation>(activation);
        layer.filters = read_size();
        layer.kernel_size = read_size();
        layer.stride = read_size();
        layer.padding = read_size();
        layer.pool_size = read_size();
        layer.pool_stride = read_size();
        layer.units = read_size();
        layer.residual = read_size() != 0;
        layers.push_back(layer);
    }

    // The constructor re-derives every shape, so a corrupted architecture is rejected here.
    ConvolutionalNetwork network{input, std::move(layers), static_cast<Loss>(loss_value)};

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
